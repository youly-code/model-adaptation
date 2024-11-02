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
from huggingface_hub import HfFolder, HfApi

dotenv.load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

alpaca_prompt = """### Instruction:
{0}

### Input:

### Response:
{1}"""

HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
wandb.login(key=WANDB_API_KEY)


def initialize_tokenizer(model_name: str, hf_token: str) -> AutoTokenizer:
    """Initialize and configure the tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        add_eos_token=True,
        add_bos_token=True,
        token=hf_token,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def prepare_fine_tuning():
    """Setup for model fine-tuning with improved configuration"""
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16  # Added for better precision
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
    
    # Use consolidated tokenizer initialization
    tokenizer = initialize_tokenizer(model_name, HF_TOKEN)
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Updated LoRA configuration to properly handle embeddings
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "down_proj", "up_proj",
        ],
        lora_dropout=0.15,
        bias="none",
        task_type="CAUSAL_LM",
        fan_in_fan_out=False,
        modules_to_save=["embed_tokens", "lm_head"]  # This is sufficient to handle embeddings
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
    MIN_WORD_COUNT = 10    
    MAX_WORD_COUNT = 256    
    MIN_COMPLEXITY = 0.2   
    MAX_COMPLEXITY = 1.0   
    MIN_SENTIMENT = -1.0  
    MAX_SENTIMENT = 0.2   
    
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
    
    # Use full validation set
    dataset = dataset["train"]
    print(f"Dataset size: {len(dataset)}")
    filtered_dataset = dataset.filter(filter_quality)
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    split_dataset = filtered_dataset.train_test_split(test_size=0.05, seed=42)
    
    def prepare_prompt(example):
        """Create simplified prompt structure"""
        instruction = f"Tell me about {example['topic']}"
        output = example['output']
        
        return {
            "instruction": instruction,
            "output": output
        }
    
    train_dataset = split_dataset["train"].map(prepare_prompt)
    eval_dataset = split_dataset["test"].map(prepare_prompt)

    # Print first example before tokenization
    first_example = train_dataset[0]
    formatted_prompt = alpaca_prompt.format(
        first_example["instruction"],
        first_example["output"]
    )
    print("\nFirst training prompt:")
    print(formatted_prompt)

    def tokenize_function(examples):
        """Tokenize using simplified template with proper labels"""
        texts = [
            alpaca_prompt.format(
                examples["instruction"][i],
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


def train_model(model, tokenizer, train_dataset, eval_dataset):
    """Training with proper configuration for stable generation"""
    training_args = TrainingArguments(
        output_dir="./complaint_model",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        eval_steps=1000,
        max_steps=2000,
        evaluation_strategy="steps",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.5,
        eval_accumulation_steps=2,
        fp16=True,
        optim="adamw_8bit",
        max_eval_samples=100,
        learning_rate=5e-5,
        warmup_ratio=0.2,
        weight_decay=0.02,
        load_best_model_at_end=True,
        early_stopping_patience=3,
        early_stopping_threshold=0.01,
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[ComplaintTestingCallback(test_prompts, tokenizer, every_n_steps=100)],
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
    """Generate responses with simplified prompting"""
    try:
        device = model.device
        
        # Format prompt using simplified template
        formatted_prompt = alpaca_prompt.format(
            prompt,
            ""  # Empty response section for generation
        )
        
        model_inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_token_type_ids=False
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                eos_token_id=tokenizer.eos_token_id,
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
            
            # Limit dataset sizes for testing
            # train_dataset = train_dataset.select(range(1000))
            # eval_dataset = eval_dataset.select(range(100))

            print(
                f"\nTraining with {len(train_dataset)} training samples and {len(eval_dataset)} eval samples"
            )

            # Create a single function for training arguments
            def get_training_args(run_name: str = None) -> TrainingArguments:
                """Create standardized training arguments"""
                return TrainingArguments(
                    output_dir="./complaint_model",
                    run_name=run_name or f"complaint-training-{wandb.util.generate_id()}",
                    num_train_epochs=2,
                    per_device_train_batch_size=16,
                    per_device_eval_batch_size=1,
                    gradient_accumulation_steps=8,
                    learning_rate=5e-5,
                    warmup_ratio=0.2,
                    weight_decay=0.05,
                    logging_steps=10,
                    eval_strategy="steps",
                    eval_steps=100,
                    save_strategy="steps",
                    save_steps=100,
                    max_grad_norm=1.0,
                    lr_scheduler_type="cosine_with_restarts",
                    gradient_checkpointing=True,
                    fp16=True,
                    optim="adamw_8bit",
                    group_by_length=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                    dataloader_num_workers=2,
                    dataloader_pin_memory=True,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False,
                    eval_accumulation_steps=2,
                )

            # Simple test prompts
            test_prompts = [
                "Tell me about your morning coffee",
                "How's the weather?",
                "Tell me about your neighbor's habits",
                "What's your opinion on modern workplace culture?",
            ]

            trainer = CustomTrainer(
                model=model,
                args=get_training_args(),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                callbacks=[
                    ComplaintTestingCallback(test_prompts, tokenizer, every_n_steps=100),
                    EarlyStoppingCallback(early_stopping_patience=3)
                ],
            )

            print("\nStarting training... This may take a while. 😁")
            trainer.train()

            # Updated saving section
            print("\nSaving adapter to Hugging Face Hub...")
            model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
            adapter_name = f"Llama-3.2-1B-Instruct-complaint-adapter"
            repo_id = f"leonvanbokhorst/{adapter_name}"

            # First save locally
            local_save_dir = f"./complaint_model/{adapter_name}"
            model.save_pretrained(
                local_save_dir,
                safe_serialization=True,  # Use safetensors format
            )
            
            # Save tokenizer configuration locally
            tokenizer.save_pretrained(local_save_dir)

            # Create repository
            api = HfApi()
            try:
                api.create_repo(
                    repo_id=repo_id,
                    exist_ok=True,
                    token=HF_TOKEN
                )
            except Exception as e:
                print(f"Repository creation error (may already exist): {e}")

            # Upload all files from local directory
            api.upload_folder(
                folder_path=local_save_dir,
                repo_id=repo_id,
                repo_type="model",
                token=HF_TOKEN
            )

            print(f"\nAdapter saved to: https://huggingface.co/{repo_id}")

        finally:
            wandb.finish()
