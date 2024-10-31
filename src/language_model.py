from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List
import aiohttp
import json
import logging
from enum import Enum
import openai
import ollama

# Configure logging
logger = logging.getLogger(__name__)


# Custom exception hierarchy for specific error handling
class ModelError(Exception):
    """Base exception for model-related errors."""

    pass


class ConnectionError(ModelError):
    """Raised when connection to model service fails."""

    pass


class ResponseError(ModelError):
    """Raised when model response is invalid."""

    pass


class ModelDefaults:
    """Default configuration values for language models."""

    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_MIN_WAIT_SECONDS = 1
    DEFAULT_MAX_WAIT_SECONDS = 10
    DEFAULT_TIMEOUT_SECONDS = 30


class APIEndpoints:
    """API endpoint configurations."""

    OLLAMA_DEFAULT_URL = "http://localhost:11434"
    OLLAMA_GENERATE_PATH = "/api/generate"


class ModelNames:
    """Default model names."""

    OLLAMA_DEFAULT = "llama3.2:latest"
    OPENAI_DEFAULT = "gpt-4o-mini"


class BaseLanguageModel(ABC):
    """Abstract base class defining the interface for language model implementations.

    Provides common retry configuration and response validation functionality.
    Subclasses must implement generate_response().
    """

    def __init__(self):
        # Remove retry configuration
        pass

    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate a response from the language model.

        Args:
            prompt: Input text to generate response from
            temperature: Controls randomness (0.0-1.0)
            max_tokens: Maximum length of generated response
            **kwargs: Additional model-specific parameters

        Returns:
            str: Generated response text

        Raises:
            ModelError: Base class for all model-related errors
        """
        pass

    async def _validate_response(self, response: str) -> bool:
        """Validate model response."""
        if not response or not isinstance(response, str):
            raise ResponseError("Invalid response format")
        return True


class OllamaModel(BaseLanguageModel):
    """Implementation for local Ollama API integration using official client library."""

    def __init__(
        self,
        model_name: str = ModelNames.OLLAMA_DEFAULT,
        host: str = APIEndpoints.OLLAMA_DEFAULT_URL,
    ):
        super().__init__()
        self.model_name = model_name
        self.host = host
        self.client = None

    async def generate_response(
        self,
        prompt: str,
        temperature: float = ModelDefaults.DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate response using Ollama client library."""
        try:
            logger.debug(f"Sending request to Ollama: {prompt}")
            
            if self.client is None:
                self.client = ollama.Client(host=self.host)
            
            options = {}
            if temperature is not None:
                options["temperature"] = temperature
            if max_tokens is not None:
                options["num_predict"] = max_tokens
            
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options=options,
                **kwargs,
            )
            
            if hasattr(response, '__await__'):
                response = await response
            
            response_text = response["response"]
            
            if not response_text or not isinstance(response_text, str):
                raise ResponseError("Invalid response format")
            
            return response_text

        except ollama.ResponseError as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise ResponseError(f"Ollama API error: {str(e)}") from e
        except ResponseError as e:
            logger.error(f"Invalid response: {str(e)}")
            raise  # Re-raise ResponseError directly
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise ConnectionError(f"Failed to connect to Ollama API: {str(e)}") from e


