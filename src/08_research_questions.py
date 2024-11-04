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
    title="Research Questions API",
    description="LLM-based research question generator using Ollama",
    version="1.0.0",
)


class ResearchInput(BaseModel):
    """Request model for research question generation."""

    topic: str


class ResearchResponse(BaseModel):
    """Response model for research questions."""

    main_question: str
    sub_questions: list[str]


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
        pass


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
            prompt = f"""As an applied research methodology expert, generate one main research question 
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
                model="hermes3", messages=[{"role": "user", "content": prompt}]
            )

            content = response["message"]["content"]

            try:
                lines = [line.strip() for line in content.strip().split("\n")]
                main_question = next(
                    line.split(":", 1)[1].strip()
                    for line in lines
                    if line.startswith("Main Question:")
                )

                sub_questions = [
                    line.split(".", 1)[1].strip()
                    for line in lines
                    if line.strip() and line[0].isdigit()
                ]

            except (IndexError, StopIteration) as e:
                logger.error(f"Failed to parse LLM response: {content}")
                raise ValueError(f"Invalid response format: {str(e)}") from e

            return {"main_question": main_question, "sub_questions": sub_questions}

        except Exception as e:
            logger.error(f"Error generating research questions: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error generating research questions: {str(e)}"
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
        raise HTTPException(status_code=500, detail="Internal server error") from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "08_research_questions:app",
        host="0.0.0.0",
        port=8001,  # Note: Changed port to avoid conflict
        log_level="info",
        reload=True,
    )

"""USAGE:
curl -X POST "http://localhost:8001/research" -H "Content-Type: application/json" -d '{"topic": "Impact of social media on mental health"}'
"""
