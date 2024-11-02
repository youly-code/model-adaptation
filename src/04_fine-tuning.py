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
    EarlyStoppingCallback,
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
import bitsandbytes as bnb
from huggingface_hub import HfFolder

dotenv.load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

alpaca_prompt = """### Instruction:
{0}

### Input:
{1}

### Response:
{2}"""

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
    """Setup for lower memory usage"""
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16  
    )

    # Initialize model with stricter memory constraints
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
    )
    
    # Initialize tokenizer with chat template
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Adjusted LoRA configuration
    lora_config = LoraConfig(
        r=32,  # Reduced from 64 for better memory efficiency
        lora_alpha=32,  # Increased from 16 for stronger adaptation
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],  # Added more target modules
        lora_dropout=0.1,  # Increased dropout for regularization
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

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


def filter_quality(example: Dict[str, Any]) -> bool:
    """Filter dataset examples based on quality criteria.
    
    Args:
        example: Dataset example containing 'output' text and quality metrics
        
    Returns:
        bool: True if example meets quality criteria
    """
    text = example['output']
    
    # Extract metrics with more lenient default values
    complexity = example.get('complexity', 0.5)  # Default to middle value
    sentiment = example.get('sentiment', -0.5)   # Default to moderately negative
    word_count = len(text.split())
    
    # More lenient quality criteria thresholds
    MIN_WORD_COUNT = 20    
    MAX_WORD_COUNT = 256    
    MIN_COMPLEXITY = 0.4   
    MAX_COMPLEXITY = 1.0   
    MIN_SENTIMENT = -0.9  
    MAX_SENTIMENT = -0.1   
    
    # Basic length check
    if not (MIN_WORD_COUNT <= word_count <= MAX_WORD_COUNT):
        return False
        
    # More lenient complexity and sentiment checks
    if complexity is not None and not (MIN_COMPLEXITY <= complexity <= MAX_COMPLEXITY):
        return False
        
    if sentiment is not None and not (MIN_SENTIMENT <= sentiment <= MAX_SENTIMENT):
        return False
    
    # Simple text quality check - avoid excessive special characters
    special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,!?]', text)) / len(text)
    if special_char_ratio > 0.2:  # Increased from 0.1
        return False
        
    return True


def prepare_dataset(tokenizer):
    """Prepare dataset with Alpaca-style prompt template"""
    dataset = load_dataset("leonvanbokhorst/synthetic-complaints-v2")
    
    # Reduce validation set size
    dataset = dataset["train"]
    print(f"Dataset size: {len(dataset)}")
    filtered_dataset = dataset.filter(filter_quality)
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    split_dataset = filtered_dataset.train_test_split(test_size=0.05, seed=42)
    
    def prepare_prompt(example):
        """Create Alpaca-style prompt structure"""
        instruction = "You are a complaining assistant."
        input_text = f"Tell me about {example['topic']}"
        output = example['output']
        
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output
        }
    
    train_dataset = split_dataset["train"].map(prepare_prompt)
    eval_dataset = split_dataset["test"].select(range(10)).map(prepare_prompt)

    # Print first example before tokenization
    first_example = train_dataset[0]
    formatted_prompt = alpaca_prompt.format(
        first_example["instruction"],
        first_example["input"],
        first_example["output"]
    )
    print("\nFirst training prompt:")
    print(formatted_prompt)

    def tokenize_function(examples):
        """Tokenize using Alpaca template with proper labels"""
        texts = [
            alpaca_prompt.format(
                examples["instruction"][i],
                examples["input"][i],
                examples["output"][i]
            ) + tokenizer.eos_token
            for i in range(len(examples["instruction"]))
        ]
        
        # Tokenize with proper padding and labels
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        
        # Create labels by copying input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        # Mask labels before the response section (we only want to train on the response)
        for idx, text in enumerate(texts):
            # Find the position of "### Response:" in the tokenized input
            response_token_ids = tokenizer.encode("### Response:")
            input_ids = tokenized["input_ids"][idx].tolist()
            
            # Find the start of the response section
            for i in range(len(input_ids) - len(response_token_ids)):
                if input_ids[i:i+len(response_token_ids)] == response_token_ids:
                    # Mask everything before the response with -100
                    tokenized["labels"][idx, :i+len(response_token_ids)] = -100
                    break
        
        return tokenized

    tokenized_train = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=train_dataset.column_names
    )

    tokenized_eval = eval_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=eval_dataset.column_names
    )

    print(f"Training samples: {len(tokenized_train)}")
    print(f"Evaluation samples: {len(tokenized_eval)}")

    return tokenized_train, tokenized_eval


def calculate_negativity(text: str) -> float:
    """Calculate negativity score using VADER sentiment analysis"""
    try:
        # Ensure VADER lexicon is downloaded
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)

        # Initialize VADER
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)
        
        # Convert compound score to 0-1 range where 1 is most negative
        # VADER compound score is between -1 (most negative) and 1 (most positive)
        # We convert it to 0 (most positive) to 1 (most negative)
        negativity = (1 - scores["compound"]) / 2
        
        return negativity

    except Exception as e:
        print(f"Error calculating negativity: {e}")
        return 0.5  # Return neutral score on error


def compute_metrics(eval_preds) -> Dict[str, float]:
    """Compute custom metrics for complaint quality with safer token handling"""
    predictions, labels = eval_preds
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    try:
        # Process in batches
        decoded_preds = []
        for pred in predictions:
            try:
                # Ensure pred is a 1D array/list of tokens
                if isinstance(pred[0], (list, np.ndarray)):
                    pred = pred[0]  # Take first sequence if nested
                
                # Filter invalid token IDs
                valid_tokens = [
                    int(token) for token in pred  # Convert to int
                    if isinstance(token, (int, np.integer)) and  # Ensure it's a number
                    0 <= token < tokenizer.vocab_size  # Check range
                ]
                
                # Decode filtered tokens
                if valid_tokens:
                    text = tokenizer.decode(valid_tokens, skip_special_tokens=True)
                    decoded_preds.append(text)
                else:
                    decoded_preds.append("")
                    
            except Exception as e:
                print(f"Decoding error for single prediction: {e}")
                decoded_preds.append("")

        # Calculate metrics
        negativity_scores = []
        for text in decoded_preds:
            if text.strip():  # Only process non-empty strings
                try:
                    score = calculate_negativity(text)
                    if isinstance(score, (float, int)):
                        negativity_scores.append(score)
                except Exception as e:
                    print(f"Scoring error: {e}")

        # Return average scores, defaulting to 0 if no valid scores
        avg_negativity = float(np.mean(negativity_scores)) if negativity_scores else 0.0
        return {
            "negativity": avg_negativity,
            "valid_samples": len(negativity_scores)
        }

    except Exception as e:
        print(f"Metrics computation error: {e}")
        return {
            "negativity": 0.0,
            "valid_samples": 0
        }



def train_model(model, tokenizer, train_dataset, eval_dataset):
    """Training with proper configuration for stable generation"""
    training_args = TrainingArguments(
        output_dir="./complaint_model",
        num_train_epochs=2,
        # Reduce validation batch size and frequency
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,  # Smaller eval batch size
        gradient_accumulation_steps=2,
        eval_steps=500,  # Increase steps between validations
        max_steps=2000,
        evaluation_strategy="steps",
        # Memory optimizations
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, 
        max_grad_norm=0.5,
        # Evaluation optimizations
        eval_accumulation_steps=4,  # Add this to accumulate eval batches
        # Conservative learning settings
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.02,
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Use eval loss to determine best model
        greater_is_better=False,  # Lower loss is better
        early_stopping_patience=3,  # Stop if no improvement for 3 evaluation rounds
        early_stopping_threshold=0.01,  # Minimum change to qualify as an improvement
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
    """Context manager for inference with CUDA optimization"""
    training_state = model.training
    cache_state = model.config.use_cache

    # Use CUDA for WSL2 + NVIDIA setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    try:
        model.eval()
        model.config.use_cache = True
        yield
    finally:
        model.train(training_state)
        model.config.use_cache = cache_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def inference_example(model, tokenizer, prompt: str) -> str:
    """Generate responses with Alpaca-style prompting"""
    try:
        device = model.device
        
        # Format prompt using Alpaca template
        formatted_prompt = alpaca_prompt.format(
            "You are a complaining assistant.",
            prompt,
            ""  # Empty response section for generation
        )
        
        model_inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            return_token_type_ids=False
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                early_stopping=True,
                repetition_penalty=1.3,
                no_repeat_ngram_size=4,
                eos_token_id=tokenizer.eos_token_id,
                length_penalty=1.0,
            )

        # Update response extraction
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part after "### Response:"
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        return response

    except Exception as e:
        print(f"Generation error: {str(e)}")
        return f"Generation failed: {str(e)}"


class CustomTrainer(Trainer):
    """Custom trainer with proper gradient handling"""

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        """Override training step with proper gradient scaling"""
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            # Get loss directly from compute_loss
            loss = self.compute_loss(model, inputs)
            
            # Ensure we have a tensor
            if isinstance(loss, dict):
                loss = loss['loss']
            elif hasattr(loss, 'loss'):
                loss = loss.loss
            
            if not isinstance(loss, torch.Tensor):
                raise ValueError(f"Expected loss to be a tensor, got {type(loss)}")

            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            if self.args.fp16 or self.args.bf16:
                self.accelerator.backward(loss)
            else:
                loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with proper handling of model outputs"""
        if "labels" not in inputs:
            raise ValueError("Labels not found in inputs")
            
        outputs = model(**inputs)
        
        # The loss should be directly available in the outputs
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
                f"\nTraining with {len(train_dataset)} training samples and {len(eval_dataset)} eval samples"
            )

            # Optimized training arguments
            training_args = TrainingArguments(
                output_dir="./complaint_model",
                run_name=f"complaint-training-{wandb.util.generate_id()}",
                num_train_epochs=3,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                gradient_accumulation_steps=4,
                learning_rate=1e-5,
                warmup_ratio=0.1,
                weight_decay=0.05,
                logging_steps=10,
                evaluation_strategy="steps",
                eval_steps=100,
                save_strategy="steps",
                save_steps=100,
                max_grad_norm=1.0,
                lr_scheduler_type="cosine_with_restarts",
                gradient_checkpointing=True,
                fp16=True,
                optim="adamw_torch",
                group_by_length=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                dataloader_num_workers=4,
                dataloader_pin_memory=True,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
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
                    ComplaintTestingCallback(test_prompts, tokenizer, every_n_steps=100),
                    EarlyStoppingCallback(early_stopping_patience=3)
                ],
            )

            print("\nStarting training... This may take a while. 😁")
            trainer.train()

            # Add this section to save the adapter
            print("\nSaving adapter to Hugging Face Hub...")
            model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
            adapter_name = f"complaint-adapter-{wandb.util.generate_id()}"
            repo_id = f"leonvanbokhorst/{adapter_name}"

            # Save adapter weights and config
            model.save_pretrained(
                f"./complaint_model/{adapter_name}",
                push_to_hub=True,
                use_auth_token=HF_TOKEN,
                repo_id=repo_id
            )
            
            # Save tokenizer
            tokenizer.save_pretrained(
                f"./complaint_model/{adapter_name}",
                push_to_hub=True,
                use_auth_token=HF_TOKEN,
                repo_id=repo_id
            )

            print(f"\nAdapter saved to: https://huggingface.co/{repo_id}")

        finally:
            wandb.finish()
