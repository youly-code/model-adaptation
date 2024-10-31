import pytest
from src.language_model import OllamaModel, ResponseError, ConnectionError

# Add integration marker
pytestmark = pytest.mark.integration

@pytest.fixture
async def ollama_model():
    """Fixture to create OllamaModel instance."""
    model = OllamaModel(model_name="llama3.2:latest")
    yield model
    # Cleanup if needed

@pytest.mark.asyncio
async def test_basic_generation(ollama_model):
    """Test basic response generation."""
    prompt = "What is 2+2?"
    response = await ollama_model.generate_response(prompt)

    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.asyncio
async def test_temperature_parameter(ollama_model):
    """Test generation with different temperature settings."""
    prompt = "Write a short poem"
    response1 = await ollama_model.generate_response(prompt, temperature=0.1)
    response2 = await ollama_model.generate_response(prompt, temperature=1.0)

    assert isinstance(response1, str)
    assert isinstance(response2, str)

@pytest.mark.asyncio
async def test_invalid_model_name():
    """Test handling of invalid model name."""
    model = OllamaModel(model_name="nonexistent_model")

    with pytest.raises(ResponseError):
        await model.generate_response("Test prompt")
