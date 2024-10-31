import json
from typing import List, Dict
import random
from datetime import datetime
import ollama
from tqdm import tqdm  # Add this import for progress bars
import csv
from pathlib import Path
import datasets  # Add this import
from datetime import timezone
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
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Literal

dotenv.load_dotenv()

# Initialize Ollama client
client = ollama.Client()

# Constants
USE_OLLAMA = True
OLLAMA_MODEL = "hermes3:latest"

COMPLAINT_STYLES = [
    "frustrated",
    "angry",
    "disappointed",
    "concerned",
    "upset",
    "irritated",
    "annoyed",
    "dissatisfied",
    "worried",
    "outraged",
    "exasperated",
    "fed up",
    "discouraged",
    "troubled",
    "displeased",
    "impatient",
    "aggravated",
    "distressed",
    "unhappy",
    "disturbed",
    "bored",
    "annoyed",
    "sarcastic",
    "ironic",
    "bitter",
    "resentful",
    "disheartened",
    "disillusioned",
    "dismayed",
    "dismissive",
    "dispassionate",
]

COMPLAINT_TOPICS = [
    "work-life balance",
    "commute",
    "workplace culture",
    "salary",
    "job stress",
    "management",
    "workload",
    "office environment",
    "career growth",
    "coworkers",
    "noise",
    "healthcare",
    "mental health",
    "wellbeing",
    "family",
    "friends",
    "pets",
    "roommates",
    "neighbors",
    "public transportation",
    "environment",
    "politics",
    "shopping",
    "food",
    "weather",
    "technology",
    "security",
    "privacy",
    "internet",
    "smartphones",
    "travel",
    "events",
    "entertainment",
    "sports",
    "insurance",
    "legal issues",
    "home repairs",
    "personal finance",
    "small talk",
    "social media",
    "dating",
    "relationships",
    "childcare",
    "welfare",
    "taxes",
    "homeownership",
    "renting",
    "real estate",
    "neighborhood",
    "environmental issues",
    "wildlife",
    "climate change",
    "recycling",
    "waste management",
    "animals",
    "art",
    "music",
    "books",
    "movies",
    "television",
    "yourself as a person",
    "yourself as a professional",
    "yourself as a parent",
    "yourself as a partner",
    "yourself as a friend",
    "yourself as a child",
    "your life",
    "your experiences",
    "your thoughts",
    "your feelings",
    "your relationships",
    "your goals",
    "your dreams",
    "your struggles",
    "your successes",
    "the past",
    "the future",
]

# Add new constants for instruction tuning
INSTRUCTION_TEMPLATE = """Below is a complaint about {topic}. Write a {style} response.

Complaint: {complaint}

Response: """

SYSTEM_PROMPT = "You are an AI assistant that helps users express their complaints in different emotional styles."


# Add this class at the top with other imports
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        return obj.isoformat() if isinstance(obj, datetime) else super().default(obj)


class GenerationConfig(BaseModel):
    batch_size: int = 50
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: int = 1000
    quality_threshold: float = 0.6

    class Config:
        extra = "allow"


def calculate_complexity(text: str) -> float:
    """Calculate linguistic complexity score of text."""
    try:
        # Ensure NLTK packages are downloaded
        nltk.download("punkt", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)

        # Tokenize text
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)

        # Calculate various complexity metrics
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        pos_tags = nltk.pos_tag(words)

        # Count complex structures
        complex_words = sum(len(word) > 6 for word in words)
        subordinate_conjunctions = sum(tag in ["IN"] for _, tag in pos_tags)

        # Compute normalized complexity score
        complexity = (
            0.3 * (avg_sentence_length / 20)  # Normalize by typical sentence length
            + 0.4 * (complex_words / len(words))  # Ratio of complex words
            + 0.3
            * (subordinate_conjunctions / len(words))  # Ratio of complex structures
        )

        return min(max(complexity, 0.0), 1.0)  # Ensure score is between 0 and 1

    except Exception as e:
        logging.warning(f"Error calculating complexity: {e}")
        return 0.0


def contains_inappropriate_content(text: str) -> bool:
    """Check if text contains inappropriate or harmful content.

    Args:
        text: Input text to check

    Returns:
        bool: True if inappropriate content detected
    """
    # Load profanity words list (you might want to maintain this in a separate file)
    inappropriate_patterns = [
        r"\b(hate|kill)\b",  # Violence-related
        r"\b(discriminatory|racist)\b",  # Discrimination-related
    ]

    text_lower = text.lower()

    # Check for inappropriate patterns
    for pattern in inappropriate_patterns:
        if nltk.re.search(pattern, text_lower):
            return True

    # Check sentiment for extremely negative content
    blob = TextBlob(text)
    return blob.sentiment.polarity < -0.8


async def get_random_style() -> str:
    """Get a random complaint style from predefined options."""
    return random.choice(COMPLAINT_STYLES)


async def get_random_topic() -> str:
    """Get a random complaint topic from predefined options."""
    return random.choice(COMPLAINT_TOPICS)  # Get one random topic from the fixed list


def analyze_response(text: str) -> dict:
    """Analyze generated text for various metrics."""
    blob = TextBlob(text)
    return {
        "sentiment": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
        "word_count": len(text.split()),
        "complexity_score": calculate_complexity(text),
    }


def quality_filter(data_point: Dict) -> bool:
    """Filter out low-quality or inappropriate responses."""
    try:
        response = data_point["output"]
        analysis = analyze_response(response)
        return (
            analysis["word_count"] >= 10
            and analysis["complexity_score"] > 0.3
            and not contains_inappropriate_content(response)
        )
    except (KeyError, AttributeError) as e:
        logging.warning(f"Error in quality filter: {e}")
        return False


class DatasetManager:
    def __init__(self, hf_repo: str):
        self.hf_repo = hf_repo
        self.api = HfApi()

    def deduplicate(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """Remove duplicate entries based on response similarity.

        Args:
            dataset: Input dataset to deduplicate

        Returns:
            datasets.Dataset: Deduplicated dataset
        """
        # Convert responses to TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words="english")
        response_vectors = vectorizer.fit_transform(
            [item["response"] for item in dataset]
        )

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(response_vectors)

        # Track indices to keep
        indices_to_keep = []
        seen = set()

        for idx in range(len(dataset)):
            if idx in seen:
                continue

            # Find similar responses
            similar_indices = np.where(similarity_matrix[idx] > 0.85)[0]

            # Keep only the first occurrence
            indices_to_keep.append(idx)
            seen.update(similar_indices)

        # Create new dataset with unique entries
        return dataset.select(indices_to_keep)

    def balance_dataset(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """Ensure balanced distribution of styles and topics."""
        # Implementation for dataset balancing

    def version_control(self, dataset: datasets.Dataset, version: str):
        """Maintain dataset versions with metadata."""
        metadata = {
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stats": self.calculate_stats(dataset),
        }
        # Push with version tracking


def log_generation_error(error: Exception, params: dict):
    """Log errors that occur during generation."""
    logging.error(
        {
            "error": str(error),
            "params": params,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


def monitor_generation(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            log_generation_metrics(duration, len(result), kwargs.get("style"))
            return result
        except Exception as e:
            log_generation_error(e, kwargs)
            raise

    return wrapper


def log_generation_metrics(duration: float, count: int, style: dict):
    logging.info(
        {
            "duration": duration,
            "samples_generated": count,
            "style": style,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


def generate_complex_prompt(topic: str, style_hash: str) -> str:
    """Generate a complex prompt combining topic and style.

    Args:
        topic: The complaint topic
        style_hash: Hashed style identifier

    Returns:
        str: Generated prompt
    """
    return f"Respond in a {style_hash} way: You complain about {topic}."


@lru_cache(maxsize=1000)
def get_cached_prompt(topic: str, style_hash: str) -> str:
    """Cache frequently used prompt combinations."""
    return generate_complex_prompt(topic, style_hash)


async def generate_batch(prompts: List[str]) -> List[str]:
    """Parallel generation of multiple samples."""
    tasks = [
        asyncio.create_task(
            client.generate(
                model=OLLAMA_MODEL,
                prompt=prompt,
            )
        )
        for prompt in prompts
    ]
    return await asyncio.gather(*tasks)


def sample_topics(n_samples: int) -> List[str]:
    """Sample topics randomly from COMPLAINT_TOPICS."""
    return random.choices(COMPLAINT_TOPICS, k=n_samples)


def sample_styles(n_samples: int) -> List[str]:
    """Sample styles randomly from COMPLAINT_STYLES."""
    return random.choices(COMPLAINT_STYLES, k=n_samples)


async def generate_synthetic_data(
    n_samples: int = 100,
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> List[Dict]:
    """Generate synthetic data in instruction tuning format."""
    synthetic_data = []

    topics = sample_topics(n_samples)
    styles = sample_styles(n_samples)

    for i in range(n_samples):
        try:
            topic = topics[i % len(topics)]
            style = styles[i % len(styles)]

            instruction = f"Respond with only one sentence. Write a {style} complaint about {topic}."

            response = client.generate(
                model=OLLAMA_MODEL,
                prompt=instruction,
            )

            cleaned_response = "".join(
                c for c in response["response"] if c.isalnum() or c in " .,!?-_'"
            )

            # Simplified data point structure
            data_point = {
                "instruction": instruction,
                "input": "",  # Empty for this use case
                "output": cleaned_response,
                "metadata": {
                    **analyze_response(cleaned_response),
                    "style": style,
                    "topic": topic,
                },
            }

            # Use the quality filter directly on the response
            if len(
                cleaned_response.split()
            ) >= 10 and not contains_inappropriate_content(cleaned_response):
                synthetic_data.append(data_point)

        except Exception as e:
            print(f"Error generating sample: {str(e)}")
            continue

    return synthetic_data


def save_synthetic_data(
    data: List[Dict],
    push_to_hf: bool = True,
    hf_repo: str = None,
    test_size: float = 0.2,
    seed: int = 42,
    max_retries: int = 3,
):
    """Save synthetic data to Hugging Face by appending to existing dataset."""
    if not push_to_hf:
        return
    if not hf_repo:
        raise ValueError("hf_repo must be specified when push_to_hf is True")

    # Flatten the data structure
    flattened_data = []
    for item in data:
        flattened_item = {
            "instruction": item["instruction"],
            "output": item["output"],
            "sentiment": item["metadata"]["sentiment"],
            "subjectivity": item["metadata"]["subjectivity"],
            "word_count": item["metadata"]["word_count"],
            "complexity_score": item["metadata"]["complexity_score"],
            "style": item["metadata"]["style"],
            "topic": item["metadata"]["topic"],
        }
        flattened_data.append(flattened_item)

    for attempt in range(max_retries):
        try:
            # Load existing dataset
            try:
                existing_dataset = datasets.load_dataset(hf_repo)
            except Exception:
                # If dataset doesn't exist, create new one
                existing_dataset = datasets.DatasetDict({
                    "train": datasets.Dataset.from_list([]),
                    "test": datasets.Dataset.from_list([])
                })

            # Convert new data to Dataset
            new_dataset = datasets.Dataset.from_list(flattened_data)
            splits = new_dataset.train_test_split(test_size=test_size, seed=seed)

            # Concatenate with existing data
            merged_dataset = datasets.DatasetDict({
                "train": datasets.concatenate_datasets([existing_dataset["train"], splits["train"]]),
                "test": datasets.concatenate_datasets([existing_dataset["test"], splits["test"]])
            })

            # Push to hub
            merged_dataset.push_to_hub(
                hf_repo,
                private=False,
                token=os.environ["HF_TOKEN"],
                max_shard_size="500MB",
                embed_external_files=False,
            )
            print(f"Successfully appended dataset on attempt {attempt + 1}")
            break

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to upload after {max_retries} attempts. Final error: {str(e)}")
                raise
            print(f"Attempt {attempt + 1} failed, retrying... Error: {str(e)}")
            time.sleep(5 * (attempt + 1))  # Exponential backoff


async def main():
    for i in range(100):
        # Generate a larger initial dataset
        all_synthetic_data = []
        for _ in tqdm(range(10), desc=f"Generating batch {i + 1} of 100"):
            batch_data = await generate_synthetic_data(
                n_samples=10
            )  # 10 samples per batch
            all_synthetic_data.extend(batch_data)

        save_synthetic_data(
            all_synthetic_data,
            push_to_hf=True,
            hf_repo="leonvanbokhorst/synthetic-complaints-v2",
            max_retries=3,
        )


if __name__ == "__main__":
    asyncio.run(main())
