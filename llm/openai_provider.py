"""OpenAI provider implementation."""
from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from .base_provider import BaseLLMProvider

T = TypeVar('T', bound=BaseModel)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider implementation."""
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        model_name: str,
        timeout: int = 120,
        retries: int = 3,
        backoff_min: float = 2.0,
        backoff_max: float = 8.0,
        reasoning_effort: Optional[str] = None,
        **kwargs
    ):
        """Initialize OpenAI provider."""
        self.config = config
        self.model_name = model_name
        self.timeout = timeout
        self.retries = retries
        self.backoff_min = backoff_min
        self.backoff_max = backoff_max
        self.reasoning_effort = reasoning_effort
        # Verbose logging toggle (suppress request logs by default)
        logging_cfg = config.get("logging", {}) if isinstance(config, dict) else {}
        env_verbose = os.environ.get("HOUND_LLM_VERBOSE", "").lower() in {"1","true","yes","on"}
        self.verbose = bool(logging_cfg.get("llm_verbose", False) or env_verbose)
        
        # Get API key from environment
        api_key_env = config.get("openai", {}).get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {api_key_env}")
        
        self.client = OpenAI(api_key=api_key)
    
    def parse(self, *, system: str, user: str, schema: Type[T]) -> T:
        """Make a structured call using OpenAI's beta.chat.completions.parse."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        
        # Log request details
        request_chars = len(system) + len(user)
        if self.verbose:
            print(f"\n[OpenAI Request]")
            print(f"  Model: {self.model_name}")
            print(f"  Schema: {schema.__name__}")
            print(f"  Total prompt: {request_chars:,} chars (~{request_chars//4:,} tokens)")
        
        last_err = None
        
        for attempt in range(self.retries):
            try:
                attempt_start = time.time()
                if self.verbose:
                    print(f"  Attempt {attempt + 1}/{self.retries}...")
                
                # Use OpenAI's structured output
                completion = self.client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=messages,
                    response_format=schema,
                    timeout=self.timeout
                )
                
                # Log response details
                response_time = time.time() - attempt_start
                response_content = completion.choices[0].message.content or ""
                if self.verbose:
                    print(f"  Response in {response_time:.2f}s ({len(response_content):,} chars)")
                    if hasattr(completion, 'usage'):
                        print(f"  Tokens: {completion.usage.total_tokens}")
                
                # Get the parsed response
                if completion.choices[0].message.parsed:
                    return completion.choices[0].message.parsed
                elif completion.choices[0].message.refusal:
                    raise RuntimeError(f"Model refused: {completion.choices[0].message.refusal}")
                else:
                    # Fallback to manual parsing
                    json_str = completion.choices[0].message.content
                    return schema.model_validate_json(json_str)
                    
            except Exception as e:
                last_err = e
                if self.verbose:
                    print(f"  Error: {e}")
                if attempt < self.retries - 1:
                    sleep_time = random.uniform(self.backoff_min, self.backoff_max)
                    if self.verbose:
                        print(f"  Retrying after {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
        
        raise RuntimeError(f"OpenAI call failed after {self.retries} attempts: {last_err}")
    
    def raw(self, *, system: str, user: str) -> str:
        """Make a plain text call."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        
        last_err = None
        for attempt in range(self.retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    timeout=self.timeout
                )
                return completion.choices[0].message.content
                
            except Exception as e:
                last_err = e
                if attempt < self.retries - 1:
                    sleep_time = random.uniform(self.backoff_min, self.backoff_max)
                    time.sleep(sleep_time)
        
        raise RuntimeError(f"OpenAI raw call failed after {self.retries} attempts: {last_err}")
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "OpenAI"
    
    @property
    def supports_thinking(self) -> bool:
        """OpenAI models may support reasoning effort but not explicit thinking mode."""
        return False
