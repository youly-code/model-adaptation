import json
from typing import List, Dict
import random
from datetime import datetime
import ollama
from tqdm import tqdm  # Add this import for progress bars
import csv
from pathlib import Path
import datasets  # Add this import
from huggingface_hub import HfApi  # Add this import
import os
import dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings  # Updated import
import uuid
from typing import Optional
import nltk
from textblob import TextBlob
import logging
from typing import Callable
from functools import wraps
import time
from functools import lru_cache
import asyncio

dotenv.load_dotenv()

# Initialize Ollama client
client = ollama.Client()

# Constants
USE_OLLAMA = True
OLLAMA_MODEL = "mistral-nemo:latest"

# Add this near the top of the file with other class definitions
class ComplaintData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    topic: str
    style: dict
    response: str
    metadata: dict = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "topic": "job",
                "style": {"emotional_state": "angry", "communication_style": "formal"},
                "response": "...",
                "metadata": {"toxicity_score": 0.3, "sentiment": -0.8}
            }
        }

def generate_synthetic_data(
    prompt: str,
    n_samples: int = 100,
    style: str = "default",
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> List[Dict]:
    """Generate synthetic data using an LLM.

    Args:
        prompt: Base prompt template for data generation
        n_samples: Number of samples to generate
        style: Style/tone for generation (e.g. formal, casual)
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens per response

    Returns:
        List of dictionaries containing generated data
    """
    synthetic_data = []

    # Add style guidance to prompt
    style_prompt = f"Use a {style} style/tone. Respond with one sentence. {prompt}"

    # Generate samples with progress bar
    for _ in tqdm(range(n_samples), desc="Generating samples"):
        try:
            # Generate response using Ollama
            response = client.generate(
                model=OLLAMA_MODEL,
                prompt=style_prompt,
                # temperature=temperature,
                # max_tokens=max_tokens,
            )

            # Parse response into structured data
            data_point = {
                "style": style,
                "instruction": prompt,
                "response": "".join(
                    c for c in response["response"] if c.isalnum() or c in " .,!?-_'"
                ),
            }

            synthetic_data.append(data_point)

        except Exception as e:
            print(f"Error generating sample: {str(e)}")
            continue

    return synthetic_data


def save_synthetic_data(
    data: List[Dict], 
    output_file: str, 
    push_to_hf: bool = True, 
    hf_repo: str = None,
    test_size: float = 0.2,  # 20% for testing by default
    seed: int = 42
):
    """Save synthetic data to file and optionally push to Hugging Face with train/test splits.

    Args:
        data: List of data dictionaries
        output_file: Path to output file (.json or .csv)
        push_to_hf: Whether to push the dataset to Hugging Face
        hf_repo: Hugging Face repository name (e.g. 'username/repo-name')
        test_size: Fraction of data to use for testing (default: 0.2)
        seed: Random seed for reproducibility
    """
    output_path = Path(output_file)

    # Save locally first
    if output_path.suffix == ".json":
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    elif output_path.suffix == ".csv":
        if not data:
            return
        fieldnames = data[0].keys()
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    else:
        raise ValueError("Output file must be .json or .csv")

    # Push to Hugging Face if requested
    if push_to_hf:
        if not hf_repo:
            raise ValueError("hf_repo must be specified when push_to_hf is True")

        try:
            # Try to load existing dataset
            existing_dataset = datasets.load_dataset(hf_repo)
            existing_train = existing_dataset['train']
            existing_test = existing_dataset['test']
            
            # Convert new data to Dataset and split
            new_dataset = datasets.Dataset.from_list(data)
            new_splits = new_dataset.train_test_split(
                test_size=test_size, 
                seed=seed
            )
            
            # Combine with existing splits
            combined_train = datasets.concatenate_datasets([
                existing_train, 
                new_splits['train']
            ])
            combined_test = datasets.concatenate_datasets([
                existing_test, 
                new_splits['test']
            ])
            
        except Exception as e:
            print(f"No existing dataset found or error occurred: {e}")
            # If loading fails, create new dataset and split
            dataset = datasets.Dataset.from_list(data)
            splits = dataset.train_test_split(
                test_size=test_size, 
                seed=seed
            )
            combined_train = splits['train']
            combined_test = splits['test']

        # Create DatasetDict with both splits
        combined_dataset = datasets.DatasetDict({
            'train': combined_train,
            'test': combined_test
        })

        # Push to hub
        combined_dataset.push_to_hub(
            hf_repo, 
            private=False, 
            token=os.environ["HF_TOKEN"]
        )


def get_random_style():
    return random.choice(
        [
            "sarcastic",
            "boring",
            "confused",
            "angry",
            "cynical",
            "dismayed",
            "dismissive",
            "disinterested",
            "displeased",
            "disrespected",
            "disregarded",
            "manipulative",
            "downtrodden",
            "anxious",
            "apathetic",
            "shy",
            "nervous",
            "guilty",
            "upset",
            "sad",
            "depressed",
            "irritated",
            "frustrated",
            "disheartened",
        ]
    )


def get_random_topic():
    return random.choice(
        [
            "your future",
            "your past",
            "your house",
            "your family",
            "your friends",
            "your job",
            "your relationship",
            "your finances",
            "your safety",
            "your privacy",
            "your reputation",
            "your mental health",
            "your physical health",
            "your appearance",
            "your intelligence",
            "your abilities",
            "your achievements",
            "your failures",
        ]
    )


def get_enhanced_styles():
    return {
        "emotional_states": [
            "sarcastic", "angry", "anxious", "depressed",
            # ... existing emotions ...
        ],
        "communication_styles": [
            "formal", "casual", "professional", "academic",
            "confrontational", "passive-aggressive"
        ],
        "personality_types": [
            "entitled", "perfectionist", "people-pleaser",
            "analytical", "impulsive", "cautious"
        ]
    }


def generate_complex_prompt(topic: str, style: dict) -> str:
    """Generate more nuanced prompts combining multiple factors."""
    return f"""
    Respond as someone who is {style['emotional_states']} and typically {style['personality_types']},
    using a {style['communication_styles']} tone.
    You complain about {topic}, expressing specific details and consequences.
    """


def analyze_response(text: str) -> dict:
    """Analyze generated text for various metrics."""
    blob = TextBlob(text)
    return {
        "sentiment": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
        "word_count": len(text.split()),
        "complexity_score": calculate_complexity(text)
    }


def quality_filter(data_point: ComplaintData) -> bool:
    """Filter out low-quality or inappropriate responses."""
    analysis = analyze_response(data_point.response)
    return (
        analysis["word_count"] >= 10 and
        analysis["complexity_score"] > 0.3 and
        not contains_inappropriate_content(data_point.response)
    )


class DatasetManager:
    def __init__(self, hf_repo: str):
        self.hf_repo = hf_repo
        self.api = HfApi()
        
    def deduplicate(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """Remove duplicate entries based on response similarity."""
        # Implementation using text similarity metrics
        
    def balance_dataset(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """Ensure balanced distribution of styles and topics."""
        # Implementation for dataset balancing
        
    def version_control(self, dataset: datasets.Dataset, version: str):
        """Maintain dataset versions with metadata."""
        metadata = {
            "version": version,
            "timestamp": datetime.utcnow().isoformat(),
            "stats": self.calculate_stats(dataset)
        }
        # Push with version tracking


def monitor_generation(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            log_generation_metrics(duration, len(result), kwargs.get('style'))
            return result
        except Exception as e:
            log_generation_error(e, kwargs)
            raise
    return wrapper

def log_generation_metrics(duration: float, count: int, style: dict):
    logging.info({
        "duration": duration,
        "samples_generated": count,
        "style": style,
        "timestamp": datetime.utcnow().isoformat()
    })


@lru_cache(maxsize=1000)
def get_cached_prompt(topic: str, style_hash: str) -> str:
    """Cache frequently used prompt combinations."""
    return generate_complex_prompt(topic, style_hash)

async def generate_batch(prompts: List[str]) -> List[str]:
    """Parallel generation of multiple samples."""
    tasks = [
        asyncio.create_task(client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
        ))
        for prompt in prompts
    ]
    return await asyncio.gather(*tasks)

def main():
    # Initialize list to store all synthetic data
    all_synthetic_data = []

    # Generate data in batches
    for _ in range(10):
        prompt = f"You complain about {get_random_topic()}."
        synthetic_data = generate_synthetic_data(
            prompt, n_samples=random.randint(1, 5), style=get_random_style()
        )
        # Extend the all_synthetic_data list with new batch
        all_synthetic_data.extend(synthetic_data)
        print(synthetic_data[0]["response"])

    # Save all accumulated data at once
    save_synthetic_data(
        all_synthetic_data,
        "synthetic_data_complete.json",
        push_to_hf=True,
        hf_repo="leonvanbokhorst/synthetic-complaints",
    )


if __name__ == "__main__":
    main()

class GenerationConfig(BaseModel):
    ollama_model: str = "mistral-nemo:latest"
    batch_size: int = 50
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: int = 1000
    quality_threshold: float = 0.6
    
    class Config:
        extra = "allow"
