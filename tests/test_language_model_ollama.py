import pytest
from unittest.mock import Mock, patch
from src.language_model import OllamaModel, ResponseError, ConnectionError
import ollama

@pytest.fixture
def mock_ollama_client():
    with patch('ollama.Client') as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        yield mock_instance

@pytest.mark.asyncio
async def test_generate_response_success(mock_ollama_client):
    expected_response = "Test response"
    mock_ollama_client.generate.return_value = {"response": expected_response}
    
    model = OllamaModel()
    response = await model.generate_response("Test prompt")
    
    assert response == expected_response
    mock_ollama_client.generate.assert_called_once_with(
        model="llama3.2:latest",
        prompt="Test prompt",
        options={"temperature": 0.7}
    )

@pytest.mark.asyncio
async def test_generate_response_with_params(mock_ollama_client):
    expected_response = "Test response"
    mock_ollama_client.generate.return_value = {"response": expected_response}
    
    model = OllamaModel()
    await model.generate_response(
        prompt="Test prompt",
        temperature=0.5,
        max_tokens=100
    )
    
    mock_ollama_client.generate.assert_called_once_with(
        model="llama3.2:latest",
        prompt="Test prompt",
        options={"temperature": 0.5, "num_predict": 100}
    )

@pytest.mark.asyncio
async def test_generate_response_api_error(mock_ollama_client):
    class MockResponseError(Exception):
        pass
    mock_ollama_client.generate.side_effect = MockResponseError("API Error")
    
    model = OllamaModel()
    with pytest.raises(ResponseError, match="Ollama API error: API Error"):
        await model.generate_response("Test prompt")

@pytest.mark.asyncio
async def test_generate_response_connection_error(mock_ollama_client):
    mock_ollama_client.generate.side_effect = Exception("Connection failed")
    
    model = OllamaModel()
    with pytest.raises(ConnectionError, match="Failed to connect to Ollama API: Connection failed"):
        await model.generate_response("Test prompt")

@pytest.mark.asyncio
async def test_generate_response_invalid_response(mock_ollama_client):
    mock_ollama_client.generate.return_value = {"response": None}
    
    model = OllamaModel()
    with pytest.raises(ResponseError, match="Invalid response format"):
        await model.generate_response("Test prompt")