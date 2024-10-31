import pytest
from unittest.mock import AsyncMock, patch, Mock
import ollama
from src.language_model import OllamaModel, ResponseError, ConnectionError, ModelDefaults

@pytest.fixture
def ollama_model():
    """Create an OllamaModel instance for testing."""
    return OllamaModel()

@pytest.fixture
def mock_ollama_client():
    """Mock the Ollama client responses."""
    with patch('ollama.Client') as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.generate = Mock()
        yield mock_instance

@pytest.mark.asyncio
async def test_generate_response_success(ollama_model, mock_ollama_client):
    """Test successful response generation."""
    expected_response = "Test response"
    mock_ollama_client.generate.return_value = {"response": expected_response}

    response = await ollama_model.generate_response("Test prompt")
    
    assert response == expected_response
    mock_ollama_client.generate.assert_called_once_with(
        model=ollama_model.model_name,
        prompt="Test prompt",
        options={"temperature": ModelDefaults.DEFAULT_TEMPERATURE}
    )

@pytest.mark.asyncio
async def test_generate_response_with_params(ollama_model, mock_ollama_client):
    """Test response generation with custom parameters."""
    mock_ollama_client.generate.return_value = {"response": "Test response"}

    await ollama_model.generate_response(
        prompt="Test prompt",
        temperature=0.5,
        max_tokens=100
    )

    mock_ollama_client.generate.assert_called_once_with(
        model=ollama_model.model_name,
        prompt="Test prompt",
        options={"temperature": 0.5, "num_predict": 100}
    )

@pytest.mark.asyncio
async def test_generate_response_api_error(ollama_model, mock_ollama_client):
    """Test handling of API errors."""
    mock_ollama_client.generate.side_effect = ollama.ResponseError("API Error")

    with pytest.raises(ResponseError, match="Ollama API error: API Error"):
        await ollama_model.generate_response("Test prompt")

@pytest.mark.asyncio
async def test_generate_response_connection_error(ollama_model, mock_ollama_client):
    """Test handling of connection errors."""
    mock_ollama_client.generate.side_effect = Exception("Connection failed")

    with pytest.raises(ConnectionError, match="Failed to connect to Ollama API: Connection failed"):
        await ollama_model.generate_response("Test prompt")

@pytest.mark.asyncio
async def test_generate_response_invalid_response(ollama_model, mock_ollama_client):
    """Test handling of invalid response format."""
    mock_ollama_client.generate.return_value = {"response": None}

    with pytest.raises(ResponseError, match="Invalid response format"):
        await ollama_model.generate_response("Test prompt")