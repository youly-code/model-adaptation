import json
from typing import List, Dict
import random
from datetime import datetime
import ollama
from tqdm import tqdm  # Add this import for progress bars
import csv
from pathlib import Path

# Initialize Ollama client
client = ollama.Client()

# Constants
USE_OLLAMA = True
OLLAMA_MODEL = "mistral-nemo:latest"  # or your preferred model


class AIEthicsJokeGenerator:
    def __init__(self):
        # Core components for joke generation
        self.setups = [
            "Why don't AI ethics professors trust GPT models?",
            "What did the AI ethics professor say to the neural network?",
            "How many AI ethics professors does it take to debug a biased model?",
            "What's an AI ethics professor's favorite breakfast?",
            "Why did the AI ethics professor get kicked out of the machine learning conference?",
            "What's an AI ethics professor's favorite exercise?",
            "How does an AI ethics professor debug their code?",
            "What's an AI ethics professor's favorite movie?",
            "Why did the AI ethics professor bring a ladder to the deep learning lecture?",
            "What's an AI ethics professor's favorite dance move?",
            "What's an AI ethics professor's favorite way to teach?",
            "What's an AI ethics professor's favorite way to debug?",
        ]

        self.punchlines = [
            "Because they keep generating their own terms and conditions!",
            "You need to work on your BIAS-ceps!",
            "None, they form a committee to discuss the societal implications first!",
            "ETHICS-press-o with a side of FAIR-trade coffee!",
            "They kept asking the robots for informed consent!",
            "Running ethical implications!",
            "They don't - they form a focus group to ensure participatory design!",
            "The Bias Identity!",
            "To help students reach a higher moral ground!",
            "The responsible AI shuffle!",
            "To get to the ethical side of the road!",
        ]

        self.ai_concepts = [
            "bias",
            "fairness",
            "transparency",
            "accountability",
            "privacy",
            "consent",
            "automation",
            "responsibility",
            "ethics",
            "discrimination",
            "love",
            "explainability",
            "transparency",
            "correctness",
            "doom",
            "apocalypse",
            "ai-armageddon",
            "ai-apocalypse",
        ]

        self.difficulty_levels = ["beginner", "intermediate", "advanced", "master"]

        self.joke_types = ["pun", "wordplay", "situational", "meta", "dad joke"]

        self.contexts = [
            "lecture",
            "research",
            "conference",
            "lab",
            "office hours",
            "peer review",
            "workshop",
            "meeting",
            "seminar",
            "holiday",
            "divorce",
            "wedding",
            "birthday",
            "funeral",
            "party",
        ]

    def generate_joke(self) -> Dict:
        """Generate a single structured joke with metadata."""
        setup = random.choice(self.setups)
        punchline = random.choice(self.punchlines)

        # Ensure some correlation between setup and punchline
        if "debug" in setup.lower():
            punchline = self.punchlines[6]
        elif "breakfast" in setup.lower():
            punchline = self.punchlines[3]

        # Use Ollama to enhance the joke (optional)
        if random.random() < 0.3:  # 30% chance to enhance
            prompt = f"Make this joke funnier while keeping the same theme:\n{setup}\n{punchline}"
            response = ollama.chat(
                model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}]
            )
            enhanced_joke = response["message"]["content"].split("\n")
            if len(enhanced_joke) >= 2:
                setup, punchline = enhanced_joke[0], enhanced_joke[1]

        return {
            "setup": setup,
            "punchline": punchline,
            "concepts": random.sample(self.ai_concepts, k=random.randint(1, 3)),
            "difficulty": random.choice(self.difficulty_levels),
            "type": random.choice(self.joke_types),
            "context": random.choice(self.contexts),
            "metadata": {
                "created": datetime.now().isoformat(),
                "version": "1.0",
                "language": "en",
            },
        }

    def generate_custom_joke(self, concept: str) -> Dict:
        """Generate a joke about a specific AI ethics concept."""
        # Use Ollama to generate a custom joke
        prompt = f"""Generate a short, funny joke about AI ethics focusing on the concept of {concept}.
        Format:
        Setup: [your setup]
        Punchline: [your punchline]
        
        Make it clever and appropriate for an academic setting."""

        response = ollama.chat(
            model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}]
        )

        joke_text = response["message"]["content"]

        # Parse the response or fall back to templates
        try:
            setup = joke_text.split("Setup: ")[1].split("\n")[0].strip()
            punchline = joke_text.split("Punchline: ")[1].split("\n")[0].strip()
        except IndexError:
            # Fall back to template jokes if parsing fails
            templates = {
                "bias": (
                    "Why did the biased AI become a comedian?",
                    "It only knew one-sided jokes!",
                ),
                "privacy": (
                    "What's a privacy-conscious AI's favorite game?",
                    "Hide and Don't Seek!",
                ),
                "fairness": (
                    "Why was the fairness algorithm feeling down?",
                    "It couldn't find its bias-life balance!",
                ),
                "transparency": (
                    "What's a transparent AI's least favorite clothing?",
                    "Black boxes!",
                ),
                "accountability": (
                    "Why did the accountability AI start a blog?",
                    "To keep a public LOG of its decisions!",
                ),
            }
            setup, punchline = templates.get(
                concept,
                (
                    f"Why did the {concept} AI cross the road?",
                    "To get to the ethical side!",
                ),
            )

        return {
            "setup": setup,
            "punchline": punchline,
            "concepts": [concept],
            "difficulty": "intermediate",
            "type": "concept_specific",
            "context": "teaching",
            "metadata": {
                "created": datetime.now().isoformat(),
                "version": "1.0",
                "language": "en",
            },
        }

    def generate_dataset(self, num_jokes: int = 100) -> List[Dict]:
        """Generate a dataset of jokes with progress tracking."""
        print("\nGenerating random jokes...")

        dataset = [
            self.generate_joke()
            for _ in tqdm(
                range(num_jokes - len(self.ai_concepts)), desc="Random jokes"
            )
        ]
        print("\nGenerating concept-specific jokes...")
        # Generate concept-specific jokes with progress bar
        dataset.extend(
            self.generate_custom_joke(concept)
            for concept in tqdm(self.ai_concepts, desc="Concept jokes")
        )
        return dataset


def create_training_data():
    """Create and save the training dataset with progress tracking."""
    print("\nInitializing AI Ethics Joke Generator...")
    generator = AIEthicsJokeGenerator()

    print("\nGenerating dataset...")
    dataset = generator.generate_dataset(1000)

    # Ensure data directory exists
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    # Define CSV paths
    raw_jokes_path = data_dir / "ai_ethics_jokes_raw.csv"
    training_data_path = data_dir / "ai_ethics_jokes_training.csv"

    # Save raw jokes to CSV
    print("\nSaving raw jokes to CSV...")
    is_new_file = not raw_jokes_path.exists()

    with open(raw_jokes_path, "a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "timestamp",
            "setup",
            "punchline",
            "concepts",
            "difficulty",
            "type",
            "context",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if is_new_file:
            writer.writeheader()

        for joke in tqdm(dataset, desc="Saving raw jokes"):
            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(),
                    "setup": joke["setup"],
                    "punchline": joke["punchline"],
                    "concepts": ",".join(joke["concepts"]),
                    "difficulty": joke["difficulty"],
                    "type": joke["type"],
                    "context": joke["context"],
                }
            )

    # Convert to training format and save
    print("\nConverting to training format and saving...")
    is_new_training_file = not training_data_path.exists()

    with open(training_data_path, "a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "timestamp",
            "instruction",
            "input",
            "output",
            "difficulty",
            "type",
            "context",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if is_new_training_file:
            writer.writeheader()

        for joke in tqdm(dataset, desc="Saving training data"):
            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(),
                    "instruction": "Generate a funny joke about AI ethics and society.",
                    "input": f"Topic: {', '.join(joke['concepts'])}",
                    "output": f"{joke['setup']}\n{joke['punchline']}",
                    "difficulty": joke["difficulty"],
                    "type": joke["type"],
                    "context": joke["context"],
                }
            )

    print(f"\nDataset generation complete! Generated and saved {len(dataset)} jokes.")
    print(f"Raw jokes saved to: {raw_jokes_path}")
    print(f"Training data saved to: {training_data_path}")

    return dataset


if __name__ == "__main__":
    dataset = create_training_data()

    print("\nExample Jokes:")
    for i, joke in enumerate(random.sample(dataset, 5)):
        print(f"\nJoke {i+1}:")
        print(f"Setup: {joke['setup']}")
        print(f"Punchline: {joke['punchline']}")
        print(f"Concepts: {', '.join(joke['concepts'])}")
        print(f"Type: {joke['type']}")
        print(f"Context: {joke['context']}")
