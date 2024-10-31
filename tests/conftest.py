import sys
import os
import pytest
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for tests."""
    logging.basicConfig(level=logging.DEBUG)

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Set up environment variables for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
