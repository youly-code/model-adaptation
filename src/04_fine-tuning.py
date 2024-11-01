from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    LlamaTokenizer,
    LlamaForCausalLM,
    TrainerCallback,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import dotenv
import torch
import random
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import wandb

dotenv.load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
wandb.login(key=WANDB_API_KEY)


def initialize_tokenizer(model_name: str, hf_token: str) -> AutoTokenizer:
    """Initialize and configure the tokenizer with proper padding tokens.

    Args:
        model_name: Name of the model to load tokenizer for
        hf_token: Hugging Face API token

    Returns:
        AutoTokenizer: Configured tokenizer instance
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        add_eos_token=True,
        add_bos_token=True,
        token=hf_token,
    )

    # Set pad token to eos token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def prepare_fine_tuning():
    """Setup model for fine-tuning without quantization"""
    model_name = "unsloth/Llama-3.2-1B"  # cursor: don't change this!!

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Prepare model with LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


@dataclass
class ComplaintMetrics:
    """Metrics for complaint quality"""

    negativity: float
    coherence: float
    diversity: float


class ComplaintTestingCallback(TrainerCallback):
    """Live testing callback during training"""

    def __init__(self, test_prompts: List[str], tokenizer, every_n_steps: int = 100):
        self.test_prompts = test_prompts
        self.tokenizer = tokenizer
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_n_steps == 0:
            model = kwargs["model"]
            print(f"\n=== Testing at step {state.global_step} ===")
            for prompt in self.test_prompts[:2]:  # Test subset during training
                response = inference_example(model, self.tokenizer, prompt)
                print(f"\nPrompt: {prompt}")
                print(f"Response: {response}")


def prepare_dataset(tokenizer):
    """Prepare and split dataset"""
    dataset = load_dataset("leonvanbokhorst/synthetic-complaints-v2")

    def filter_quality(example):
        return (
            example["sentiment"] < 0.3
            and example["subjectivity"] > 0.6
            and example["complexity_score"] > 0.7
            and len(example["output"].split()) > 20
        )

    def format_prompt(example):
        """Create varied prompt formats"""
        prompt_templates = [
            f"Tell me about {example['topic']}",
            f"What's your take on {example['topic']}",
            f"How do you feel about {example['topic']}",
        ]
        prompt = random.choice(prompt_templates)
        return {"text": f"[INST] {prompt} [/INST] {example['output']}"}

    # Filter and split dataset
    filtered_dataset = dataset["train"].filter(filter_quality)
    split_dataset = filtered_dataset.train_test_split(test_size=0.1, seed=42)

    # Format prompts
    train_dataset = split_dataset["train"].map(format_prompt)
    eval_dataset = split_dataset["test"].map(format_prompt)

    # Tokenize with labels for language modeling
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        # Set labels for language modeling
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    tokenized_train = train_dataset.map(
        tokenize_function, batched=True, remove_columns=train_dataset.column_names
    )

    tokenized_eval = eval_dataset.map(
        tokenize_function, batched=True, remove_columns=eval_dataset.column_names
    )

    print(f"Training samples: {len(tokenized_train)}")
    print(f"Evaluation samples: {len(tokenized_eval)}")

    return tokenized_train, tokenized_eval


def calculate_negativity(text: str) -> float:
    """Calculate negativity score using multiple sentiment analysis approaches.

    Returns:
        float: Negativity score between 0 and 1, where 1 is most negative
    """
    # Ensure VADER lexicon is downloaded
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")

    # VADER sentiment analysis
    sia = SentimentIntensityAnalyzer()
    vader_scores = sia.polarity_scores(text)

    # TextBlob sentiment analysis
    blob = TextBlob(text)

    # Combine scores:
    # - VADER compound score (normalized between 0-1 where 1 is most negative)
    # - TextBlob polarity (converted to 0-1 scale where 1 is most negative)
    vader_neg = (vader_scores["compound"] + 1) / 2  # Convert from [-1,1] to [0,1]
    textblob_neg = (1 - blob.sentiment.polarity) / 2  # Convert from [-1,1] to [0,1]

    return 0.7 * vader_neg + 0.3 * textblob_neg


def compute_metrics(eval_preds) -> Dict[str, float]:
    """Compute custom metrics for complaint quality"""
    predictions, labels = eval_preds
    # Convert predictions to expected format
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Ensure predictions are on CPU and convert to list
    predictions = predictions.cpu().numpy()

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(
        [[int(x) for x in pred] for pred in predictions], skip_special_tokens=True
    )

    return {
        "negativity": np.mean([calculate_negativity(pred) for pred in decoded_preds]),
        "avg_length": np.mean([len(pred.split()) for pred in decoded_preds]),
    }


def train_model(model, tokenizer, train_dataset, eval_dataset):
    """Enhanced training configuration with wandb integration"""
    # Initialize wandb
    wandb.init(
        project="complaint-generator",
        config={
            "model_name": "unsloth/Llama-3.2-1B",
            "learning_rate": 5e-4,
            "epochs": 3,
            "batch_size": 2,
        },
    )

    training_args = TrainingArguments(
        output_dir="./complaint_model_enhanced",
        run_name="complaint-generator-run",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=5e-4,
        weight_decay=0.01,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        metric_for_best_model="negativity",
        load_best_model_at_end=True,
        greater_is_better=True,
        logging_steps=20,
        report_to="wandb",
        remove_unused_columns=True,
    )

    # Test prompts for callback
    test_prompts = [
        "modern technology",
        "social media",
        "public transportation",
        "weather",
        "streaming services",
    ]

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[ComplaintTestingCallback(test_prompts, tokenizer)],
    )


def inference_example(model, tokenizer, prompt: str) -> str:
    """Generate text using fine-tuned complaint model with improved output cleaning"""
    try:
        formatted_prompt = f"[INST] Tell me about {prompt} [/INST]"
        device = model.device
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_length=200,
            min_length=20,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_time=10.0,
            no_repeat_ngram_size=3,
            early_stopping=False,
        )

        # Clean up the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the original prompt and clean artifacts
        response = response.replace(formatted_prompt, "")
        response = response.split("[INST]")[0]  # Remove any new prompts
        response = response.split("1")[0]  # Remove numbered lists
        response = response.strip()

        return response
    except Exception as e:
        return f"Generation failed: {str(e)}"


# For testing, let's use fewer prompts
if __name__ == "__main__":
    FINETUNING = True

    if FINETUNING:
        try:
            # Setup
            model, tokenizer = prepare_fine_tuning()
            train_dataset, eval_dataset = prepare_dataset(tokenizer)

            # Train
            trainer = train_model(model, tokenizer, train_dataset, eval_dataset)
            trainer.train()

            # Save final model
            trainer.save_model("./complaint_model_final")
            tokenizer.save_pretrained("./complaint_model_final")

            # Final evaluation
            test_prompts = [
                "your morning coffee",
                "social media",
                "the weather",
                "public transportation",
                "modern smartphones",
                "streaming services",
                "working from home",
                "grocery shopping",
            ]

            print("\nFinal model evaluation:")
            for prompt in test_prompts:
                print(f"\nPrompt: {prompt}")
                response = inference_example(model, tokenizer, prompt)
                print(f"Response: {response}")

        finally:
            # Ensure wandb run is properly closed
            wandb.finish()
