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
from contextlib import contextmanager

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
    """Setup model for fine-tuning optimized for Apple Silicon"""
    model_name = "unsloth/Llama-3.2-1B"

    try:
        # Initialize tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="right",
            add_eos_token=True,
            add_bos_token=True,
        )

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Configure model loading
        model_config = {
            "torch_dtype": torch.float32,
            "use_cache": False,
            "use_flash_attention_2": False,  # Disable for MPS compatibility
        }

        # Load model with base configuration
        model = LlamaForCausalLM.from_pretrained(model_name, **model_config)

        # Apply LoRA config before other modifications
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
        )

        # Prepare model for training
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()

        # Explicitly enable gradients for all parameters
        model.train()
        for param in model.parameters():
            param.requires_grad = True

        # Move to device after all configurations
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")
        model = model.to(device)

        # Verify gradients are enabled
        grad_params = [p.requires_grad for p in model.parameters()]
        print(f"Parameters requiring gradients: {sum(grad_params)}/{len(grad_params)}")

    except Exception as e:
        raise RuntimeError(f"Failed to initialize model/tokenizer: {str(e)}")

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
    """Memory-optimized negativity score calculation"""
    # Use only VADER for sentiment analysis (more memory efficient)
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    sia = SentimentIntensityAnalyzer()
    vader_scores = sia.polarity_scores(text)

    # Convert compound score to 0-1 range where 1 is most negative
    return (vader_scores["compound"] + 1) / 2


def compute_metrics(eval_preds) -> Dict[str, float]:
    """Compute custom metrics for complaint quality"""
    predictions, labels = eval_preds
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Ensure we're working with numpy arrays
    predictions = predictions.astype(np.int64)

    try:
        # Process in batches
        decoded_preds = []
        for pred in predictions:
            try:
                # Filter padding tokens and convert to list
                tokens = [int(t) for t in pred if t not in [-100, tokenizer.pad_token_id]]
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                decoded_preds.append(text)
            except Exception as e:
                print(f"Decoding error: {e}")
                decoded_preds.append("")

        # Calculate metrics
        negativity_scores = []
        for text in decoded_preds:
            if text.strip():  # Only process non-empty strings
                try:
                    score = calculate_negativity(text)
                    if isinstance(score, (float, int)):  # Ensure score is a scalar
                        negativity_scores.append(score)
                except Exception as e:
                    print(f"Scoring error: {e}")

        # Return average scores, defaulting to 0 if no valid scores
        avg_negativity = float(np.mean(negativity_scores)) if negativity_scores else 0.0
        return {"negativity": avg_negativity}

    except Exception as e:
        print(f"Metrics computation error: {e}")
        return {"negativity": 0.0}


def train_model(model, tokenizer, train_dataset, eval_dataset):
    """Training configuration optimized for M3"""
    wandb.init(
        project="complaint-generator",
        config={
            "model_name": "unsloth/Llama-3.2-1B",
            "learning_rate": 5e-4,
            "epochs": 3,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
        },
    )

    training_args = TrainingArguments(
        output_dir="./complaint_model_enhanced",
        run_name=f"complaint-generator-{wandb.run.id}",
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Reduced batch size
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # Increased for stability
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=1e-4,  # Reduced learning rate
        weight_decay=0.01,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        report_to="wandb",
        remove_unused_columns=True,
        fp16=False,
        bf16=False,  # Disabled both fp16 and bf16
        gradient_checkpointing=True,
    )

    # Test prompts for callback
    test_prompts = [
        "modern technology",
        "social media",
    ]  # Reduced number of test prompts

    # Create a very small evaluation dataset
    max_eval_samples = 20  # Reduced from 50
    eval_dataset = eval_dataset.select(range(min(max_eval_samples, len(eval_dataset))))

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            ComplaintTestingCallback(test_prompts, tokenizer, every_n_steps=200)
        ],  # Reduced callback frequency
    )


@contextmanager
def inference_mode(model):
    """Context manager with M3 optimization"""
    training_state = model.training
    cache_state = model.config.use_cache

    # Ensure we're using MPS if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    try:
        model.eval()
        model.config.use_cache = True
        yield
    finally:
        model.train(training_state)
        model.config.use_cache = cache_state


def inference_example(model, tokenizer, prompt: str) -> str:
    """Generate a response using the fine-tuned model."""
    try:
        device = model.device
        formatted_prompt = f"[INST] Tell me about {prompt} [/INST]"
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        with inference_mode(model):
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(formatted_prompt, "").strip()

    except Exception as e:
        print(f"Generation error: {str(e)}")
        return f"Generation failed: {str(e)}"


# For testing, let's use fewer prompts
if __name__ == "__main__":
    FINETUNING = True

    if FINETUNING:
        try:
            # Setup
            model, tokenizer = prepare_fine_tuning()
            train_dataset, eval_dataset = prepare_dataset(tokenizer)

            # Quick test setup - but with enough data to learn
            train_dataset = train_dataset.select(range(min(500, len(train_dataset))))
            eval_dataset = eval_dataset.select(range(min(25, len(eval_dataset))))

            print(
                f"Quick test with {len(train_dataset)} training samples and {len(eval_dataset)} eval samples"
            )

            # Training arguments optimized for quick testing
            training_args = TrainingArguments(
                output_dir="./complaint_model_test",
                run_name=f"complaint-quick-test-{wandb.util.generate_id()}",
                num_train_epochs=1,
                per_device_train_batch_size=1,  # Reduced batch size
                per_device_eval_batch_size=1,  # Reduced batch size
                gradient_accumulation_steps=16,  # Increased accumulation
                learning_rate=5e-6,  # Further reduced learning rate
                max_grad_norm=0.5,  # Reduced gradient clipping
                logging_steps=5,
                eval_strategy="steps",  # Changed from eval_strategy
                eval_steps=20,  # Less frequent evaluation
                save_strategy="no",
                weight_decay=0.01,
                warmup_ratio=0.1,  # Increased warmup
                report_to="wandb",
                load_best_model_at_end=False,
                # Add gradient checkpointing
                gradient_checkpointing=True,
                # Add fp16 mixed precision
                fp16=False,  # Disabled for MPS
                optim="adamw_torch",
            )

            # Simple test prompts
            test_prompts = ["modern technology", "social media"]

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                callbacks=[
                    ComplaintTestingCallback(test_prompts, tokenizer, every_n_steps=20)
                ],
            )

            print("\nStarting quick test training...")
            trainer.train()

            print("\nTesting inference:")
            test_prompt = "quick test"
            response = inference_example(model, tokenizer, test_prompt)
            print(f"Prompt: {test_prompt}")
            print(f"Response: {response}")

        finally:
            wandb.finish()
