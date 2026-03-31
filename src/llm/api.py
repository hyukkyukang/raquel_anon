import hkkang_utils.misc as misc_utils

misc_utils.load_dotenv()
import hashlib
import json
import logging
import os
import threading
from functools import cached_property
from typing import Any, Callable, List, Optional, Tuple, Union

import dspy
import hkkang_utils.pattern as pattern_utils
import litellm
import redis
from dspy.primitives import Prediction
from omegaconf import DictConfig
from openai import OpenAI

from src.llm.tracker import llm_call_tracker
from src.prompt.base import Prompt
from src.utils.env import GOOGLE_API_KEY, OPENAI_API_KEY
from src.utils.string import normalize_string

# Drop unsupported params for models like GPT-5 that don't support temperature=0.0
litellm.drop_params = True

logger = logging.getLogger("LLMAPICaller")

# Flag to ensure LiteLLM cache is only configured once
_litellm_cache_configured: bool = False


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """Safely read config values from DictConfig, dict, or namespace."""
    if cfg is None:
        return default
    if isinstance(cfg, DictConfig):
        return cfg.get(key, default)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:
            return default
    if hasattr(cfg, key):
        return getattr(cfg, key, default)
    return default


def configure_litellm_cache(global_cfg: Any) -> None:
    """Configure LiteLLM's disk cache for LLM call reproducibility.

    DSPy 2.x uses litellm internally, so we configure litellm's caching directly.
    This enables persistent disk caching for deterministic/reproducible LLM calls.

    Args:
        global_cfg: Global configuration containing llm.cache settings.
    """
    global _litellm_cache_configured
    if _litellm_cache_configured:
        return

    # Get cache config, with defaults if not specified
    llm_cfg = _cfg_get(global_cfg, "llm", {})
    cache_cfg = _cfg_get(llm_cfg, "cache", {})
    enable_disk_cache: bool = cache_cfg.get("enable_disk_cache", True)
    disk_cache_dir: Optional[str] = cache_cfg.get("disk_cache_dir", None)

    if not enable_disk_cache:
        logger.info("LiteLLM disk cache disabled by configuration")
        _litellm_cache_configured = True
        return

    try:
        # Use litellm's disk cache for persistent caching across runs
        # type="disk" provides SQLite-based persistent caching
        cache_kwargs = {"type": "disk"}
        if disk_cache_dir is not None:
            cache_kwargs["disk_cache_dir"] = disk_cache_dir

        litellm.enable_cache(**cache_kwargs)
        logger.info(
            f"LiteLLM disk cache enabled (dir={disk_cache_dir or 'default ~/.litellm/cache'})"
        )
    except Exception as e:
        logger.warning(f"Failed to configure LiteLLM cache: {e}")

    _litellm_cache_configured = True


class LLMAPICaller(metaclass=pattern_utils.SingletonMetaWithArgs):
    """A singleton class for interacting with LLM API.

    This class provides a unified interface for interacting with LLM API,
    including caching and rate limiting.

    Attributes:
        model_name: Name of the LLM model to use
        max_tokens: Maximum tokens for the response
        temperature: Sampling temperature (None = model default)
        use_custom_api: Whether to use custom API instead of DSPy
        global_cfg: Global configuration object
        seed: Random seed for deterministic output
        _cache_hit_local: Thread-local storage for tracking cache hits
    """

    # Thread-local storage for tracking cache hits across threads
    _cache_hit_local: threading.local = threading.local()

    def __init__(
        self,
        model_name: str,
        max_tokens: int,
        use_custom_api: bool,
        global_cfg: DictConfig,
        temperature: Optional[
            float
        ] = None,  # Optional: GPT-5 models only support temperature=1
        seed: Optional[int] = None,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature  # None means use model's default
        self.use_custom_api = use_custom_api
        self.global_cfg = global_cfg
        self.seed = seed  # For deterministic output
        self.__post_init__()

    def __post_init__(self):
        # Configure LiteLLM cache once (for reproducibility across runs)
        configure_litellm_cache(self.global_cfg)

        # Create and store the LM instance for this caller
        # Note: We DON'T call dspy.configure() here because it's global and would
        # be overwritten by other LLMAPICaller instances using different models
        self._dspy_lm: Optional[dspy.LM] = None
        if not self.use_custom_api:
            api_key = OPENAI_API_KEY
            if "gemini" in self.model_name.lower():
                api_key = GOOGLE_API_KEY

            # Build LM kwargs, only include temperature if specified.
            # Note: GPT-5 models only support temperature=1, so we don't pass it for them.
            # DSPy/LiteLLM parameter compatibility changes across releases; keep
            # this kwargs set to broadly supported request arguments only.
            lm_kwargs = {
                "api_key": api_key,
                "max_tokens": self.max_tokens,
            }
            if self.temperature is not None:
                lm_kwargs["temperature"] = self.temperature
            if self.seed is not None:
                lm_kwargs["seed"] = self.seed

            self._dspy_lm = dspy.LM(self.model_name, **lm_kwargs)

    @property
    def last_call_was_cached(self) -> bool:
        """Return whether the last API call was served from cache.

        This is thread-safe and returns the cache hit status for the calling thread.
        Useful for conditional rate limiting - skip rate limiting for cached responses.

        Returns:
            True if the last call was a cache hit, False otherwise.
        """
        return getattr(self._cache_hit_local, "was_cached", False)

    def _set_cache_hit(self, was_cached: bool) -> None:
        """Set the cache hit status for the current thread.

        Args:
            was_cached: Whether the last call was served from cache.
        """
        self._cache_hit_local.was_cached = was_cached

    @cached_property
    def openai_client(self) -> OpenAI:
        return OpenAI(api_key=OPENAI_API_KEY)

    @cached_property
    def redis_client(self) -> redis.Redis:
        return redis.Redis(
            host=self.global_cfg.redis.host,
            port=self.global_cfg.redis.port,
            db=self.global_cfg.redis.db,
            decode_responses=True,  # Automatically decode responses to strings
        )

    def __call__(
        self,
        prompt: Prompt,
        temperature: Optional[float] = None,
        post_process_fn: Optional[Callable[[str], Any]] = None,
        prefix: Optional[str] = None,
    ) -> Any:
        """Main method to call the OpenAI API, with optional post-processing of the result."""
        # Log the number of API CALLs
        self._log_api_call(prefix=prefix)
        if temperature is None:
            temperature = self.temperature
        result = self.call_api(
            prompt=prompt,
            temperature=temperature,
        )
        if post_process_fn is not None:
            return post_process_fn(result)
        return result

    def _log_api_call(self, prefix: Optional[str] = None) -> None:
        # Track in memory via singleton tracker
        if prefix:
            llm_call_tracker.record_call(prefix)

        # File-based logging (existing behavior)
        file_name: str = f"{prefix}_default.txt" if prefix else "default.txt"
        file_path: str = os.path.join(
            self.global_cfg.project_path, f"log/api_call/{file_name}"
        )
        # Create the file if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Read in the value if it exists, with race-condition-safe error handling
        value: int = 0
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    content: str = f.read().strip()
                    if content:
                        value = int(content)
            except (ValueError, IOError):
                # File empty/corrupted due to race condition in parallel execution
                value = 0
        # Increment the value
        value += 1
        # Write the value to the file
        with open(file_path, "w") as f:
            f.write(str(value))

    def _call_custom_api(
        self,
        system_instruction: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> str:
        """Call the OpenAI API with caching support.

        Args:
            system_instruction (str): System instruction for the model
            user_prompt (str): User prompt for the model
            temperature (float, optional): Sampling temperature. None = use model default.

        Returns:
            str: The model's response
        """
        # Only cache when temperature is explicitly 0 (deterministic)
        use_cache = temperature == 0
        if use_cache:
            try:
                # Check Redis cache for existing response
                cache_key = self.generate_cache_key(
                    model_name=self.model_name,
                    system_instruction=system_instruction,
                    user_prompt=user_prompt,
                )

                # Get cached response from Redis
                cached_response = self.redis_client.get(cache_key)

                # If cached response is found, mark as cache hit and return
                if cached_response is not None:
                    self._set_cache_hit(True)
                    return cached_response

            except redis.RedisError as e:
                logger.error(f"Redis cache error: {e}")

        # If no cache hit or Redis error, call the API
        # Mark as cache miss since we're making an actual API call
        self._set_cache_hit(False)

        # Build kwargs including seed for determinism if provided
        api_kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
        }
        # Only pass temperature if explicitly set (GPT-5 only supports temp=1)
        if temperature is not None:
            api_kwargs["temperature"] = temperature
        if self.seed is not None:
            api_kwargs["seed"] = self.seed

        response = self.openai_client.chat.completions.create(**api_kwargs)

        result = response.choices[0].message.content.strip()

        # Cache the response in Redis if temperature is 0 (deterministic)
        if use_cache:
            try:
                self.redis_client.setex(cache_key, 86400, result)  # Cache for 24 hours
            except redis.RedisError as e:
                logger.error(f"Redis cache error: {e}")

        return result

    def _call_dspy_api(self, prompt: Prompt) -> Union[List[str], str]:
        """Call DSPy API with cache hit detection.

        DSPy internally uses litellm caching. We detect cache hits by checking
        if the LM's history grows after the call. If history length is unchanged,
        the result was served from cache.

        Args:
            prompt: The prompt object to send to the LLM.

        Returns:
            The LLM response (single string or list of strings).
        """
        # Use dspy.context() for thread-safe per-call configuration.
        # This is necessary because dspy.configure() is global and NOT thread-safe,
        # causing errors when called from worker threads in parallel processing.
        # dspy.context() creates a thread-local context that works correctly with
        # ThreadPoolExecutor and async parallel processing.
        with dspy.context(lm=self._dspy_lm):
            client = dspy.ChainOfThought(prompt.signature())
            # Construct input dictionary
            input_dict = {}
            for input_field_name in prompt.signature().input_fields.keys():
                input_dict[input_field_name] = getattr(prompt, input_field_name)

            # Get the output field name (we assume there is only one output field)
            output_field_names: List[str] = list(
                prompt.signature().output_fields.keys()
            )

            # Track history length before call to detect cache hits
            # DSPy adds to history only when an actual API call is made
            history_len_before: int = len(self._dspy_lm.history) if self._dspy_lm else 0

            # Call the API
            prediction: Prediction = client(**input_dict)

            # Check if history grew (indicates actual API call, not cache hit)
            history_len_after: int = len(self._dspy_lm.history) if self._dspy_lm else 0
            was_cached: bool = history_len_after == history_len_before
            self._set_cache_hit(was_cached)

            # Get the output
            outputss: List[str] = [prediction[name] for name in output_field_names]

            if len(output_field_names) == 1:
                output: str = outputss[0]
                return output
            else:
                return outputss  # type: ignore

    def call_api(
        self,
        prompt: Prompt,
        temperature: Optional[float] = None,
    ) -> str:
        if self.use_custom_api:
            result = self._call_custom_api(
                system_instruction=prompt.system_instruction,
                user_prompt=prompt.get_user_prompt(),
                temperature=temperature,
            )
        else:
            result = self._call_dspy_api(prompt=prompt)
        return result

    def generate_cache_key(
        self, model_name: str, system_instruction: str, user_prompt: str
    ) -> str:
        """Generate a consistent cache key for the given parameters.

        Args:
            model_name (str): Name of the GPT model
            system_instruction (str): System instruction for the model
            user_prompt (str): User prompt for the model

        Returns:
            str: A unique cache key
        """
        # Normalize all input strings
        model_name = normalize_string(model_name)
        system_instruction = normalize_string(system_instruction)
        user_prompt = normalize_string(user_prompt)

        # Combine and hash
        combined = f"{model_name}:{system_instruction}:{user_prompt}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()
