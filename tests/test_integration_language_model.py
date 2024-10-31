import pytest
import pytest_asyncio
from src.language_model import OllamaModel, ResponseError, ConnectionError

# Add integration marker
pytestmark = pytest.mark.integration

@pytest.mark.asyncio
class TestOllamaModelIntegration:
    @pytest_asyncio.fixture
    async def ollama_model(self):
        """Fixture to create OllamaModel instance."""
        yield OllamaModel(model_name="llama3.2:latest")
        # Cleanup if needed

    async def test_basic_generation(self, ollama_model):
        """Test basic response generation."""
        prompt = "What is 2+2?"
        response = await ollama_model.generate_response(prompt)

        assert isinstance(response, str)
        assert len(response) > 0

    async def test_temperature_parameter(self, ollama_model):
        """Test generation with different temperature settings."""
        prompt = "Write a short poem"
        response1 = await ollama_model.generate_response(prompt, temperature=0.1)
        response2 = await ollama_model.generate_response(prompt, temperature=1.0)

        assert isinstance(response1, str)
        assert isinstance(response2, str)

    async def test_invalid_model_name(self):
        """Test handling of invalid model name."""
        model = OllamaModel(model_name="nonexistent_model")

        with pytest.raises(ResponseError):
            await model.generate_response("Test prompt")
