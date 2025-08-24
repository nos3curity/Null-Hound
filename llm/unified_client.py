"""Unified LLM client that supports multiple providers."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel

from .base_provider import BaseLLMProvider
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider
from .anthropic_provider import AnthropicProvider

T = TypeVar('T', bound=BaseModel)


class UnifiedLLMClient:
    """
    Unified LLM client that can work with multiple providers.
    
    Maintains backward compatibility with existing code while supporting
    new providers like Gemini.
    """
    
    def __init__(self, cfg: Dict[str, Any], profile: str = "graph", debug_logger=None):
        """
        Initialize unified LLM client with config and profile.
        
        The config now supports a 'provider' field for each model profile.
        If not specified, defaults to 'openai' for backward compatibility.
        
        Args:
            cfg: Configuration dictionary
            profile: Model profile to use (e.g., "graph", "agent")
            debug_logger: Optional DebugLogger instance for logging interactions
        """
        self.cfg = cfg
        self.profile = profile
        self.debug_logger = debug_logger
        
        # Get model configuration for this profile
        model_config = cfg["models"][profile]
        self.model = model_config["model"]
        
        # Determine provider (default to openai for backward compatibility)
        provider_name = model_config.get("provider", "openai").lower()
        
        # Get provider-specific configuration
        timeout_cfg = cfg.get("timeouts", {})
        retry_cfg = cfg.get("retries", {})
        
        common_kwargs = {
            "config": cfg,
            "model_name": self.model,
            "timeout": timeout_cfg.get("request_seconds", 120),
            "retries": retry_cfg.get("max_attempts", 3),
            "backoff_min": retry_cfg.get("backoff_min_seconds", 2),
            "backoff_max": retry_cfg.get("backoff_max_seconds", 8),
        }
        
        # Determine verbose logging
        logging_cfg = cfg.get("logging", {}) if isinstance(cfg, dict) else {}
        env_verbose = False
        try:
            import os as _os
            env_verbose = _os.environ.get("HOUND_LLM_VERBOSE", "").lower() in {"1","true","yes","on"}
        except Exception:
            pass
        llm_verbose = bool(logging_cfg.get("llm_verbose", False) or env_verbose)

        # Initialize the appropriate provider
        if provider_name == "openai":
            self.provider = OpenAIProvider(
                **common_kwargs,
                reasoning_effort=model_config.get("reasoning_effort"),
                verbose=llm_verbose
            )
        elif provider_name == "gemini":
            self.provider = GeminiProvider(
                **common_kwargs,
                thinking_enabled=model_config.get("thinking_enabled", False),
                thinking_budget=model_config.get("thinking_budget", -1)
            )
        elif provider_name == "anthropic":
            self.provider = AnthropicProvider(
                **common_kwargs,
                api_key_env=cfg.get("anthropic", {}).get("api_key_env", "ANTHROPIC_API_KEY"),
                verbose=llm_verbose,
                thinking_enabled=model_config.get("thinking_enabled", False)
            )
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

        # Only surface provider init when verbose explicitly enabled
        if llm_verbose:
            print(f"[*] Initialized {self.provider.provider_name} provider with model: {self.model}")
            if self.provider.supports_thinking and hasattr(self.provider, 'thinking_enabled'):
                if self.provider.thinking_enabled:
                    print(f"    Thinking mode: Enabled")
    
    def parse(self, *, system: str, user: str, schema: Type[T]) -> T:
        """
        Structured call: returns an instance of `schema` (Pydantic BaseModel).
        
        Delegates to the underlying provider.
        """
        start_time = time.time()
        error = None
        response = None
        
        try:
            response = self.provider.parse(system=system, user=user, schema=schema)
            return response
        except Exception as e:
            error = str(e)
            raise
        finally:
            # Log if debug logger is available
            if self.debug_logger:
                duration = time.time() - start_time
                self.debug_logger.log_interaction(
                    system_prompt=system,
                    user_prompt=user,
                    response=response.dict() if response and hasattr(response, 'dict') else response,
                    schema=schema,
                    duration=duration,
                    error=error
                )
    
    def raw(self, *, system: str, user: str) -> str:
        """
        Plain text call (no schema).
        
        Delegates to the underlying provider.
        """
        start_time = time.time()
        error = None
        response = None
        
        try:
            response = self.provider.raw(system=system, user=user)
            return response
        except Exception as e:
            error = str(e)
            raise
        finally:
            # Log if debug logger is available
            if self.debug_logger:
                duration = time.time() - start_time
                self.debug_logger.log_interaction(
                    system_prompt=system,
                    user_prompt=user,
                    response=response,
                    schema=None,
                    duration=duration,
                    error=error
                )
    
    def generate(self, *, system: str, user: str) -> str:
        """
        Alias for raw() - plain text generation.
        Added for compatibility with agent code.
        """
        return self.raw(system=system, user=user)
    
    @property
    def provider_name(self) -> str:
        """Get the name of the current provider."""
        return self.provider.provider_name
    
    @property
    def supports_thinking(self) -> bool:
        """Check if the current provider/model supports thinking mode."""
        return self.provider.supports_thinking


# Maintain backward compatibility by aliasing
LLMClient = UnifiedLLMClient
