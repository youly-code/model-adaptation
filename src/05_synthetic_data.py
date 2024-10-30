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

dotenv.load_dotenv()

# Initialize Ollama client
client = ollama.Client()

# Constants
USE_OLLAMA = True
OLLAMA_MODEL = "mistral-nemo:latest"

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


class ComplaintData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    instruction: str
    response: str
    metadata: dict = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "example": {
                "instruction": "Complain about your job",
                "response": "...",
                "metadata": {
                    "sentiment": -0.8,
                    "subjectivity": 0.6,
                    "word_count": 25,
                    "complexity_score": 0.7,
                },
            }
        }
    }


class GenerationConfig(BaseModel):
    ollama_model: str = "mistral-nemo:latest"
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


def quality_filter(data_point: ComplaintData) -> bool:
    """Filter out low-quality or inappropriate responses."""
    analysis = analyze_response(data_point.response)
    return (
        analysis["word_count"] >= 10
        and analysis["complexity_score"] > 0.3
        and not contains_inappropriate_content(data_point.response)
    )


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
    """Generate synthetic data using an LLM."""
    synthetic_data = []

    topics = sample_topics(n_samples)
    styles = sample_styles(n_samples)

    for i in tqdm(range(n_samples), desc="Generating samples"):
        try:
            topic = topics[i % len(topics)]
            style = styles[i % len(styles)]

            prompt = f"You complain about {topic}."
            full_prompt = f"Respond with one sentence in a {style} way: {prompt}"

            response = client.generate(
                model=OLLAMA_MODEL,
                prompt=full_prompt,
            )

            cleaned_response = "".join(
                c for c in response["response"] if c.isalnum() or c in " .,!?-_'"
            )

            data_point = {
                "instruction": prompt,
                "response": cleaned_response,
                **analyze_response(cleaned_response),
                "style": style,
                "topic": topic,
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
    seed: int = 42,
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
            existing_dataset = None
            try:
                existing_dataset = datasets.load_dataset(hf_repo)
            except Exception as e:
                print(f"No existing dataset found: {e}")
            existing_train = existing_dataset["train"]
            existing_test = existing_dataset["test"]

            # Convert new data to Dataset and split
            new_dataset = datasets.Dataset.from_list(data)
            new_splits = new_dataset.train_test_split(test_size=test_size, seed=seed)

            # Combine with existing splits
            combined_train = datasets.concatenate_datasets(
                [existing_train, new_splits["train"]]
            )
            combined_test = datasets.concatenate_datasets(
                [existing_test, new_splits["test"]]
            )

        except Exception as e:
            print(f"No existing dataset found or error occurred: {e}")
            # If loading fails, create new dataset and split
            dataset = datasets.Dataset.from_list(data)
            splits = dataset.train_test_split(test_size=test_size, seed=seed)
            combined_train = splits["train"]
            combined_test = splits["test"]

        # Create DatasetDict with both splits
        combined_dataset = datasets.DatasetDict(
            {"train": combined_train, "test": combined_test}
        )

        # Push to hub
        combined_dataset.push_to_hub(
            hf_repo, private=False, token=os.environ["HF_TOKEN"]
        )


async def main():
    # Initialize list to store all synthetic data
    all_synthetic_data = []

    # Generate data in batches
    for _ in range(100):
        batch_data = await generate_synthetic_data(n_samples=10)
        # Extend the all_synthetic_data list with new batch
        all_synthetic_data.extend(batch_data)

    # Save all accumulated data at once
    save_synthetic_data(
        all_synthetic_data,
        "synthetic_data_complete.json",
        push_to_hf=True,
        hf_repo="leonvanbokhorst/synthetic-complaints",
    )


if __name__ == "__main__":
    asyncio.run(main())
