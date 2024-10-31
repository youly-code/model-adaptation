from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama import AsyncClient
import logging
from typing import Dict
from contextlib import asynccontextmanager
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="Simple LLM-based sentiment analysis using Ollama",
    version="1.0.0"
)

class TextInput(BaseModel):
    """Request model for text input."""
    text: str

class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""
    sentiment: str
    explanation: str

class ResearchInput(BaseModel):
    """Request model for research question generation."""
    topic: str

class ResearchResponse(BaseModel):
    """Response model for research questions."""
    main_question: str
    sub_questions: list[str]

# Add configuration and client management
@lru_cache()
def get_ollama_client() -> AsyncClient:
    """Returns a cached Ollama client instance."""
    return AsyncClient()

@asynccontextmanager
async def get_client():
    """Context manager for Ollama client to ensure proper resource management."""
    client = get_ollama_client()
    try:
        yield client
    finally:
        # AsyncClient doesn't require explicit cleanup
        pass

async def analyze_sentiment(text: str) -> Dict[str, str]:
    """
    Analyzes the sentiment of given text using Ollama.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary containing sentiment and explanation
    """
    async with get_client() as client:
        try:
            prompt = f"""Analyze the sentiment of the following text and provide:
            1. A single word sentiment (positive, negative, or neutral)
            2. A brief explanation (1-2 sentences)
            
            Text: "{text}"
            
            Format your response as:
            Sentiment: [sentiment]
            Explanation: [your explanation]"""

            response = await client.chat(
                model="hermes3",
                messages=[{"role": "user", "content": prompt}]
            )

            content = response["message"]["content"]

            # Add more robust response parsing
            try:
                lines = [line.strip() for line in content.strip().split('\n')]
                sentiment = next(line.split(':', 1)[1].strip() 
                               for line in lines if line.startswith('Sentiment:'))
                explanation = next(line.split(':', 1)[1].strip() 
                                 for line in lines if line.startswith('Explanation:'))
            except (IndexError, StopIteration) as e:
                logger.error(f"Failed to parse LLM response: {content}")
                raise ValueError(f"Invalid response format: {str(e)}") from e

            return {
                "sentiment": sentiment.lower(),  # Normalize sentiment
                "explanation": explanation
            }

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error analyzing sentiment: {str(e)}"
            ) from e

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_text(input_data: TextInput) -> SentimentResponse:
    """
    Endpoint for sentiment analysis.
    
    Args:
        input_data: TextInput model containing text to analyze
        
    Returns:
        SentimentResponse with sentiment and explanation
    """
    try:
        result = await analyze_sentiment(input_data.text)
        return SentimentResponse(**result)
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def generate_research_questions(topic: str) -> Dict[str, any]:
    """
    Generates research questions for a given topic using Ollama.
    
    Args:
        topic: Input topic or idea to generate questions for
        
    Returns:
        Dictionary containing main question and sub-questions
    """
    async with get_client() as client:
        try:
            prompt = f"""As a research methodology expert, generate one main research question 
            and 3-4 supporting sub-questions for the following topic. The questions should be 
            specific, measurable, and academically rigorous.
            
            Topic: "{topic}"
            
            Format your response exactly as:
            Main Question: [your main research question]
            Sub Questions:
            1. [first sub-question]
            2. [second sub-question]
            3. [third sub-question]"""

            response = await client.chat(
                model="hermes3",
                messages=[{"role": "user", "content": prompt}]
            )

            content = response["message"]["content"]

            # Parse the response
            try:
                lines = [line.strip() for line in content.strip().split('\n')]
                main_question = next(line.split(':', 1)[1].strip() 
                                   for line in lines if line.startswith('Main Question:'))
                
                sub_questions = [
                    line.split('.', 1)[1].strip()
                    for line in lines 
                    if line.strip() and line[0].isdigit()
                ]
                
            except (IndexError, StopIteration) as e:
                logger.error(f"Failed to parse LLM response: {content}")
                raise ValueError(f"Invalid response format: {str(e)}") from e

            return {
                "main_question": main_question,
                "sub_questions": sub_questions
            }

        except Exception as e:
            logger.error(f"Error generating research questions: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error generating research questions: {str(e)}"
            ) from e

@app.post("/research", response_model=ResearchResponse)
async def generate_research(input_data: ResearchInput) -> ResearchResponse:
    """
    Endpoint for research question generation.
    
    Args:
        input_data: ResearchInput model containing topic to analyze
        
    Returns:
        ResearchResponse with main question and sub-questions
    """
    try:
        result = await generate_research_questions(input_data.topic)
        return ResearchResponse(**result)
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "07_sentiment_analysis:app",  # Use module:app format
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        reload=True
    ) 
    
    
"""USAGE:
# Original sentiment analysis
curl -X POST "http://localhost:8000/analyze" -H "Content-Type: application/json" -d '{"text": "I love this product!"}'

# New research question generator
curl -X POST "http://localhost:8000/research" -H "Content-Type: application/json" -d '{"topic": "Impact of social media on mental health"}'
"""