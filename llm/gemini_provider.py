"""Gemini provider implementation."""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any, TypeVar

import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from pydantic import BaseModel

from .base_provider import BaseLLMProvider
from .schema_definitions import get_schema_definition

T = TypeVar('T', bound=BaseModel)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider implementation."""
    
    def __init__(
        self, 
        config: dict[str, Any], 
        model_name: str,
        timeout: int = 120,
        retries: int = 3,
        backoff_min: float = 2.0,
        backoff_max: float = 8.0,
        thinking_enabled: bool = False,
        thinking_budget: int = -1,  # -1 for dynamic, 0 to disable, >0 for fixed budget
        **kwargs
    ):
        """
        Initialize Gemini provider.
        
        Args:
            config: Configuration dictionary
            model_name: Gemini model name (e.g., "gemini-2.0-flash", "gemini-2.5-flash")
            timeout: Request timeout in seconds
            retries: Number of retry attempts
            backoff_min: Minimum backoff time in seconds
            backoff_max: Maximum backoff time in seconds
            thinking_enabled: Whether to enable thinking mode (for 2.5 models)
            thinking_budget: Thinking token budget (-1 for dynamic, 0 to disable)
        """
        self.config = config
        self.model_name = model_name
        self.timeout = timeout
        self.retries = retries
        self.backoff_min = backoff_min
        self.backoff_max = backoff_max
        self.thinking_enabled = thinking_enabled
        self.thinking_budget = thinking_budget
        self._last_token_usage = None
        
        # Get API key from environment
        api_key_env = config.get("gemini", {}).get("api_key_env", "GOOGLE_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {api_key_env}")
        
        # Configure the SDK
        genai.configure(api_key=api_key)
        
        # Create the model with generation config
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            # No max_output_tokens specified - let Gemini use its maximum (8192)
            "response_mime_type": "application/json",  # For structured output
        }
        
        # Safety settings - be maximally permissive for code analysis
        # BLOCK_NONE allows all content through
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Try to add additional harm categories if they exist in the current SDK version
        try:
            if hasattr(HarmCategory, 'HARM_CATEGORY_CIVIC_INTEGRITY'):
                safety_settings[HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY] = HarmBlockThreshold.BLOCK_NONE
        except Exception:
            pass
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
    
    def parse(self, *, system: str, user: str, schema: type[T]) -> T:
        """Make a structured call using Gemini's response_schema."""
        # Get schema definition from centralized source
        schema_info = get_schema_definition(schema)
        
        # Combine system and user prompts (Gemini doesn't have separate system messages)
        prompt = f"{system}{schema_info}\n\n{user}"
        
        # Calculate request size for potential debugging
        len(prompt)
        
        last_err = None
        
        for attempt in range(self.retries):
            try:
                attempt_start = time.time()
                
                # Create generation config
                # Note: Gemini doesn't support structured output the same way as OpenAI
                # We'll ask for JSON and validate it ourselves
                generation_config = {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40,
                    # No max_output_tokens - let Gemini use its maximum
                    "response_mime_type": "application/json",
                }
                
                # Add thinking configuration if enabled and supported
                if self.thinking_enabled and "2.5" in self.model_name:
                    # For Gemini 2.5 models, we can configure thinking
                    # Note: This requires using the correct API parameters
                    # which may vary based on SDK version
                    pass  # Thinking is enabled by default in 2.5 models
                
                # Generate with structured output
                # Also apply safety settings at generation time
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=self.model._safety_settings,  # Use the model's safety settings
                    request_options={"timeout": self.timeout}
                )
                
                # Log response details
                time.time() - attempt_start
                
                # Track token usage if available
                if hasattr(response, 'usage_metadata'):
                    self._last_token_usage = {
                        'input_tokens': getattr(response.usage_metadata, 'prompt_token_count', 0),
                        'output_tokens': getattr(response.usage_metadata, 'candidates_token_count', 0),
                        'total_tokens': getattr(response.usage_metadata, 'total_token_count', 0)
                    }
                
                # Check if response was blocked
                if response.candidates:
                    candidate = response.candidates[0]
                    if candidate.finish_reason and candidate.finish_reason != 1:  # 1 = STOP (normal completion)
                        # Map finish reasons to human-readable messages (corrected mapping)
                        finish_reasons = {
                            0: "UNSPECIFIED - Reason not specified",
                            2: "MAX_TOKENS - Response exceeded maximum token limit",
                            3: "SAFETY - Response blocked by safety filters",
                            4: "RECITATION - Response blocked for recitation",
                            5: "OTHER - Response blocked for other reasons"
                        }
                        reason_msg = finish_reasons.get(candidate.finish_reason, f"Unknown reason {candidate.finish_reason}")
                        
                        # Check for safety ratings if available
                        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                            safety_info = []
                            for rating in candidate.safety_ratings:
                                if rating.probability and rating.probability > 2:  # MEDIUM or higher
                                    safety_info.append(f"{rating.category.name}: {rating.probability}")
                            if safety_info:
                                reason_msg += f" (triggered by: {', '.join(safety_info)})"
                        
                        raise RuntimeError(f"Response blocked: {reason_msg}")
                
                # Parse the JSON response into the schema
                if response.text:
                    json_data = json.loads(response.text)
                    # Debug: Log what Gemini returned (disabled)
                    # if 'graphs_needed' in json_data:
                    #     for g in json_data.get('graphs_needed', [])[:1]:
                    #         pass  # Previously logged sample graph structure
                    return schema.model_validate(json_data)
                else:
                    raise RuntimeError("Empty response from Gemini")
                    
            except Exception as e:
                last_err = e
                if attempt < self.retries - 1:
                    sleep_time = random.uniform(self.backoff_min, self.backoff_max)
                    time.sleep(sleep_time)
        
        raise RuntimeError(f"Gemini call failed after {self.retries} attempts: {last_err}")
    
    def raw(self, *, system: str, user: str) -> str:
        """Make a plain text call."""
        # Combine system and user prompts
        prompt = f"{system}\n\n{user}"
        
        last_err = None
        for attempt in range(self.retries):
            try:
                # Generate without structured output
                generation_config = {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40,
                    # No max_output_tokens - let Gemini use its maximum
                }
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=self.model._safety_settings,  # Use the model's safety settings
                    request_options={"timeout": self.timeout}
                )
                
                # Track token usage if available
                if hasattr(response, 'usage_metadata'):
                    self._last_token_usage = {
                        'input_tokens': getattr(response.usage_metadata, 'prompt_token_count', 0),
                        'output_tokens': getattr(response.usage_metadata, 'candidates_token_count', 0),
                        'total_tokens': getattr(response.usage_metadata, 'total_token_count', 0)
                    }
                
                return response.text
                
            except Exception as e:
                last_err = e
                if attempt < self.retries - 1:
                    sleep_time = random.uniform(self.backoff_min, self.backoff_max)
                    time.sleep(sleep_time)
        
        raise RuntimeError(f"Gemini raw call failed after {self.retries} attempts: {last_err}")
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "Gemini"
    
    @property
    def supports_thinking(self) -> bool:
        """Gemini 2.5 models support thinking mode."""
        return "2.5" in self.model_name
    
    def get_last_token_usage(self) -> dict[str, int] | None:
        """Return token usage from the last call if available."""
        return self._last_token_usage
