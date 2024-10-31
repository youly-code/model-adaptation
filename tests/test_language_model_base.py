import pytest
from src.language_model import BaseLanguageModel, ModelError, ResponseError
from typing import Optional

# Create a concrete test implementation of the abstract base class
class MockLanguageModel(BaseLanguageModel):
    def setup_method(self, method):
        super().__init__()
        
    async def generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs) -> str:
        return "test response"

# Test cases
@pytest.mark.asyncio
async def test_validate_response():
    model = MockLanguageModel()
    
    # Test valid response
    assert await model._validate_response("valid response") == True
    
    # Test invalid responses
    with pytest.raises(ResponseError):
        await model._validate_response(None)
    
    with pytest.raises(ResponseError):
        await model._validate_response(123)