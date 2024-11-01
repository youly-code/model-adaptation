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
import re

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
    """Setup with fixes for training stability"""
    model_name = "unsloth/Llama-3.2-1B"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        add_eos_token=True,
        add_bos_token=True,
    )

    # Ensure proper token setup
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model_config = {
        "torch_dtype": torch.float32,
        "use_cache": False,
    }

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_config)

    # More conservative LoRA config
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # Explicitly enable gradients
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    device = torch.device("mps")
    model = model.to(device)

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
    """Prepare and filter dataset with proper quality controls"""
    dataset = load_dataset("leonvanbokhorst/synthetic-complaints-v2")

    def filter_quality(example):
        """Filter for high-quality examples only"""
        text = example["output"]

        # Check for basic quality markers
        if not isinstance(text, str) or len(text.split()) < 10:
            return False

        # Ensure proper English text
        if not all(ord(char) < 128 for char in text):
            return False

        # Check if text has proper sentence structure
        if not text.rstrip()[-1] in ".!?":
            return False

        # Quality metrics from dataset
        return (
            example["sentiment"] < 0.3
            and example["subjectivity"] > 0.6
            and example["complexity_score"] > 0.7
        )

    def prepare_prompt(example):
        """Create consistent prompt structure"""
        return {
            "text": f"[INST] Tell me about {example['topic']} [/INST] {example['output']}"
        }

    # Filter and prepare data
    filtered_dataset = dataset["train"].filter(filter_quality)
    split_dataset = filtered_dataset.train_test_split(test_size=0.1, seed=42)

    train_dataset = split_dataset["train"].map(prepare_prompt)
    eval_dataset = split_dataset["test"].map(prepare_prompt)

    def tokenize_function(examples):
        """Ensure inputs are properly set up for training"""
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        # Set labels for language modeling (no gradients needed for inputs/labels)
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
                # Create boolean masks for filtering
                not_pad_mask = pred != tokenizer.pad_token_id
                not_ignore_mask = pred != -100
                valid_tokens_mask = not_pad_mask & not_ignore_mask

                # Filter tokens using the mask
                valid_tokens = pred[valid_tokens_mask].tolist()

                # Decode filtered tokens
                text = tokenizer.decode(valid_tokens, skip_special_tokens=True)
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
    """Training with proper configuration for stable generation"""
    training_args = TrainingArguments(
        output_dir="./complaint_model",
        num_train_epochs=2,
        # Smaller batches for stability
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        # Conservative learning settings
        learning_rate=5e-6,
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        weight_decay=0.02,
        # ... rest of training args ...
    )

    class QualityTestingCallback(TrainerCallback):
        """Test output quality during training"""

        def __init__(self, test_prompts, tokenizer, every_n_steps=100):
            self.test_prompts = test_prompts
            self.tokenizer = tokenizer
            self.every_n_steps = every_n_steps
            self.previous_responses = {}

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % self.every_n_steps == 0:
                model = kwargs["model"]
                model.eval()

                print(f"\n=== Testing at step {state.global_step} ===")
                for prompt in self.test_prompts:
                    response = inference_example(model, self.tokenizer, prompt)
                    print(f"\nPrompt: {prompt}")
                    print(f"Response: {response}")

                    # Track response stability
                    if prompt in self.previous_responses and response == self.previous_responses[prompt]:
                        print("Warning: Identical response to previous step")

                    self.previous_responses[prompt] = response

                model.train()

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[QualityTestingCallback(test_prompts, tokenizer, every_n_steps=100)],
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
    """Generate responses with fixed configuration"""
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

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                early_stopping=False,
                num_beams=1,
                repetition_penalty=1.2,
                bad_words_ids=[[tokenizer.encode(char)[0]] for char in "＼／＿￣#"],
                # Combined stopping criteria into single eos_token_id list
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.encode(".")[0],
                    tokenizer.encode("!")[0],
                    tokenizer.encode("?")[0],
                    tokenizer.encode("\n")[0],
                ],
            )

        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        response = tokenizer.decode(
            outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # Clean up the response
        response = response.replace(formatted_prompt, "").strip()
        response = re.sub(r"[^\x00-\x7F]+", "", response)  # Remove non-ASCII
        response = re.sub(r"#\w+", "", response)  # Remove hashtags
        response = re.sub(r"\s+", " ", response)  # Normalize whitespace

        return response

    except Exception as e:
        print(f"Generation error: {str(e)}")
        return f"Generation failed: {str(e)}"


class CustomTrainer(Trainer):
    """Custom trainer with proper gradient handling"""

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step with proper gradient handling"""
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Don't modify input tensors - let the model handle gradients
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        return loss.detach()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Compute loss with proper gradient handling"""
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


# For testing, let's use fewer prompts
if __name__ == "__main__":
    FINETUNING = True

    if FINETUNING:
        try:
            # Setup
            model, tokenizer = prepare_fine_tuning()
            train_dataset, eval_dataset = prepare_dataset(tokenizer)

            print(
                f"Training with {len(train_dataset)} training samples and {len(eval_dataset)} eval samples"
            )

            # Training arguments optimized for better learning while maintaining stability
            training_args = TrainingArguments(
                output_dir="./complaint_model",
                run_name=f"complaint-training-{wandb.util.generate_id()}",
                num_train_epochs=3,  # Increased from 1
                per_device_train_batch_size=2,  # Increased from 1
                per_device_eval_batch_size=2,  # Increased from 1
                gradient_accumulation_steps=8,  # Reduced from 16 due to larger batch size
                learning_rate=1e-5,  # Slightly increased from 5e-6
                max_grad_norm=1.0,  # Increased from 0.5
                logging_steps=20,  # Increased from 5
                eval_strategy="steps",
                eval_steps=50,  # Increased from 20
                save_strategy="steps",  # Changed from "no"
                save_steps=100,  # Added save checkpoints
                weight_decay=0.01,
                warmup_ratio=0.05,  # Reduced from 0.1
                report_to="wandb",
                load_best_model_at_end=True,  # Changed to True
                gradient_checkpointing=True,
                # M3-specific settings
                fp16=False,  # Disable mixed precision
                bf16=False,  # Disable bfloat16
                optim="adamw_torch",
            )

            # Simple test prompts
            test_prompts = ["modern technology", "social media", "cats"]

            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                callbacks=[
                    ComplaintTestingCallback(test_prompts, tokenizer, every_n_steps=20)
                ],
            )

            print("\nStarting training... This may take a while. 😁")
            trainer.train()

            print("\nTesting inference: ")
            for test_prompt in test_prompts:
                response = inference_example(model, tokenizer, test_prompt)
                print(f"Prompt: {test_prompt}")
                print(f"Response: {response}")

        finally:
            wandb.finish()
