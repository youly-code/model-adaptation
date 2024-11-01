"""
Subliminal Story Generator Module

This module generates stories with embedded subliminal messages using the Ollama LLM API.
It creates narratives that subtly incorporate specified messages and emotional themes
while maintaining the chosen genre's characteristics.

Dependencies:
    - ollama: For LLM API interaction
    - pydantic: For data validation
    - asyncio: For asynchronous operations
"""

import ollama
import random
from typing import List, Dict, Optional, Tuple
import asyncio
from pydantic import BaseModel
import logging
from functools import wraps
from datetime import datetime
import json

# Initialize Ollama client for LLM interactions
client = ollama.Client()

# Model configuration for story generation
OLLAMA_MODEL = "mistral:latest"  # Can be modified to use different models

class GenerationMetadata(BaseModel):
    """
    Detailed metadata about the story generation process.
    
    Attributes:
        prompt_analysis: Breakdown of prompt components and their purpose
        generation_parameters: LLM configuration used
        narrative_elements: Identified story elements and techniques
        subliminal_mapping: How subliminal elements were incorporated
        confidence_scores: Confidence ratings for different aspects
    """
    prompt_analysis: Dict[str, str]
    generation_parameters: Dict[str, any]
    narrative_elements: Dict[str, List[str]]
    subliminal_mapping: Dict[str, str]
    confidence_scores: Dict[str, float]
    
class StoryConfig(BaseModel):
    """
    Configuration model for story generation parameters.
    
    Attributes:
        genre (str): The literary genre of the story
        target_emotion (str): Primary emotion to evoke in readers
        subliminal_message (str): Message to embed subtly in the story
        story_length (str): Desired length of story ("short", "medium", "long")
        temperature (float): Controls randomness in generation (0.0-1.0)
        max_tokens (int): Maximum length of generated text
        explain_generation: Whether to generate detailed explainability
    """
    genre: str
    target_emotion: str
    subliminal_message: str
    story_length: str = "medium"  # short, medium, long
    temperature: float = 0.7
    max_tokens: int = 2000
    explain_generation: bool = True

# Available genres for story generation
STORY_GENRES = [
    "fairy tale", "science fiction", "mystery", 
    "adventure", "fable", "slice of life"
]

# Template for story generation prompt
STORY_TEMPLATE = """Write a {genre} story that subtly incorporates the message "{message}" 
without explicitly stating it. The story should evoke {emotion} feelings.
The story should be {length} in length.

Make the message subtle - it should influence the reader's subconscious without being obvious.
Use metaphors, symbolism, and careful word choice to convey the underlying message.

Story:"""

async def analyze_story_elements(story: str, config: StoryConfig) -> Dict[str, List[str]]:
    """Analyzes the generated story to identify key narrative elements."""
    analysis_prompt = f"""Analyze this story and identify:
    1. Key symbols and metaphors
    2. Emotional triggers
    3. Narrative techniques used
    4. Subliminal message integration points
    
    Story: {story}
    
    Provide the analysis in JSON format.
    """
    
    response = client.generate(
        model=OLLAMA_MODEL,
        prompt=analysis_prompt,
        options={"temperature": 0.2}
    )
    
    return json.loads(response["response"])

async def generate_subliminal_story(config: StoryConfig) -> Dict:
    """
    Generate a story with embedded subliminal messages and detailed explainability.
    """
    # Format the prompt template with configuration values
    prompt = STORY_TEMPLATE.format(
        genre=config.genre,
        message=config.subliminal_message,
        emotion=config.target_emotion,
        length=config.story_length
    )

    generation_metadata = GenerationMetadata(
        prompt_analysis={
            "genre_purpose": f"Using {config.genre} to establish appropriate narrative framework",
            "emotion_integration": f"Targeting {config.target_emotion} through story elements",
            "message_strategy": f"Embedding '{config.subliminal_message}' using indirect techniques"
        },
        generation_parameters={
            "model": OLLAMA_MODEL,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens
        },
        narrative_elements={},
        subliminal_mapping={},
        confidence_scores={}
    )
    
    try:
        # Generate story using Ollama API
        response = client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={
                "temperature": config.temperature,
                "num_predict": config.max_tokens
            }
        )
        
        if config.explain_generation:
            # Analyze story elements
            story_analysis = await analyze_story_elements(response["response"], config)
            generation_metadata.narrative_elements = story_analysis
            
            # Calculate confidence scores
            confidence_analysis = await client.generate(
                model=OLLAMA_MODEL,
                prompt=f"""Rate the effectiveness (0-1) of:
                1. Message subtlety
                2. Emotional impact
                3. Genre adherence
                4. Overall coherence
                
                Story: {response["response"]}
                Target message: {config.subliminal_message}
                
                Respond in JSON format.
                """,
                options={"temperature": 0.2}
            )
            
            generation_metadata.confidence_scores = json.loads(confidence_analysis["response"])
        
        return {
            "story": response["response"],
            "metadata": {
                "genre": config.genre,
                "target_emotion": config.target_emotion,
                "subliminal_message": config.subliminal_message,
                "timestamp": datetime.now().isoformat()
            },
            "explainability": generation_metadata.dict() if config.explain_generation else None
        }
        
    except Exception as e:
        logging.error(f"Story generation failed: {str(e)}")
        raise

async def main():
    """
    Example usage of the story generator.
    Creates a sample story with predefined configuration.
    """
    # Create example configuration
    config = StoryConfig(
        genre="fairy tale",
        target_emotion="wonder",
        subliminal_message="always trust your intuition",
    )
    
    # Generate and display story
    story = await generate_subliminal_story(config)
    print("\nGenerated Story:")
    print("---------------")
    print(story["story"])
    print("\nMetadata:")
    print(story["metadata"])

if __name__ == "__main__":
    asyncio.run(main()) 