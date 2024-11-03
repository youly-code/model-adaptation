from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import os
import dotenv
import torch
from typing import List, Dict, Any
import wandb
from contextlib import contextmanager
import re
from huggingface_hub import HfApi
from datasets import load_dataset

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

    # Initialize model with simpler config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        token=HF_TOKEN,
        torch_dtype=torch.float16,
    )

    # Use consolidated tokenizer initialization
    tokenizer = initialize_tokenizer(model_name, HF_TOKEN)

    # Prepare model for training (can keep this for consistency)
    model = prepare_model_for_kbit_training(model)

    # Updated LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
            "lm_head"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
        modules_to_save=["embed_tokens"],
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    return model, tokenizer


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
                if (
                    prompt in self.previous_responses
                    and response == self.previous_responses[prompt]
                ):
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
    text = example["output"]

    # Extract metrics with more lenient default values
    complexity = example.get("complexity", 0.5)  # Default to middle value
    sentiment = example.get("sentiment", -0.5)  # Default to moderately negative
    word_count = len(text.split())

    # More lenient quality criteria thresholds
    MIN_WORD_COUNT = 12
    MAX_WORD_COUNT = 256
    MIN_COMPLEXITY = 0.1
    MAX_COMPLEXITY = 1.0
    MIN_SENTIMENT = -0.9
    MAX_SENTIMENT = 0.5

    # Basic length check
    if not (MIN_WORD_COUNT <= word_count <= MAX_WORD_COUNT):
        return False

    # More lenient complexity and sentiment checks
    if complexity is not None and not (MIN_COMPLEXITY <= complexity <= MAX_COMPLEXITY):
        return False

    if sentiment is not None and not (MIN_SENTIMENT <= sentiment <= MAX_SENTIMENT):
        return False

    # Simple text quality check - avoid excessive special characters
    special_char_ratio = len(re.findall(r"[^a-zA-Z0-9\s.,!?]", text)) / len(text)
    return special_char_ratio <= 0.2


def prepare_dataset(tokenizer):
    """Prepare dataset with Alpaca-style prompt template"""
    dataset = load_dataset("leonvanbokhorst/synthetic-complaints-v2")

    # Use full validation set and shuffle
    dataset = dataset["train"].shuffle(seed=42)
    print(f"Dataset size: {len(dataset)}")
    filtered_dataset = dataset.filter(filter_quality)
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    split_dataset = filtered_dataset.train_test_split(test_size=0.05, seed=42)

    def prepare_prompt(example):
        """Create simplified prompt structure"""
        return {
            "instruction": f"Tell me about {example['topic']}", 
            "output": example["output"]
        }

    train_dataset = split_dataset["train"].map(prepare_prompt)
    eval_dataset = split_dataset["test"].map(prepare_prompt)

    def tokenize_and_add_length(examples):
        """Tokenize and add length column for group_by_length"""
        texts = [
            alpaca_prompt.format(examples["instruction"][i], examples["output"][i])
            + tokenizer.eos_token
            for i in range(len(examples["instruction"]))
        ]

        # Tokenize with proper padding and labels
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        # Create labels by copying input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()

        # Add length feature (actual sequence length before padding)
        tokenized["length"] = [
            len(tokenizer(text, truncation=False)["input_ids"])
            for text in texts
        ]

        # Mask labels before the response section
        for idx, text in enumerate(texts):
            response_token_ids = tokenizer.encode("### Response:")
            input_ids = tokenized["input_ids"][idx].tolist()

            for i in range(len(input_ids) - len(response_token_ids)):
                if input_ids[i : i + len(response_token_ids)] == response_token_ids:
                    tokenized["labels"][idx, : i + len(response_token_ids)] = -100
                    break

        return tokenized

    # Use the updated tokenization function
    tokenized_train = train_dataset.map(
        tokenize_and_add_length,
        batched=True,
        remove_columns=train_dataset.column_names
    )

    tokenized_eval = eval_dataset.map(
        tokenize_and_add_length,
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    print(f"Training samples: {len(tokenized_train)}")
    print(f"Evaluation samples: {len(tokenized_eval)}")

    return tokenized_train, tokenized_eval


def inference_example(model, tokenizer, prompt: str) -> str:
    """Generate responses with simplified prompting"""
    try:
        device = model.device

        # Format prompt using simplified template
        formatted_prompt = alpaca_prompt.format(
            prompt, ""  # Empty response section for generation
        )

        model_inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            return_token_type_ids=False,
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


def save_model(
    model, 
    tokenizer, 
    save_dir: str, 
    repo_id: str, 
    hf_token: str, 
    save_mode: str = "merged"
) -> None:
    """Save model to local directory and HuggingFace Hub.
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
        save_dir: Local directory to save files
        repo_id: HuggingFace Hub repository ID
        hf_token: HuggingFace API token
        save_mode: Either "merged" or "lora" (default: "merged")
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if save_mode == "lora":
        # Save LoRA adapter
        lora_state_dict = {
            k: v.to("cpu")
            for k, v in model.state_dict().items()
            if any(substr in k for substr in [
                "lora_A", "lora_B", 
                "lora_embedding_A", "lora_embedding_B",
                "lora_scaling"
            ])
        }
        torch.save(lora_state_dict, os.path.join(save_dir, "adapter_model.bin"))
        
        if hasattr(model, "config"):
            model.config.save_pretrained(save_dir)
        if hasattr(model, "peft_config"):
            model.peft_config["default"].save_pretrained(save_dir)
            
    else:  # merged mode
        model = model.merge_and_unload()
        model.save_pretrained(
            save_dir,
            safe_serialization=True,
            max_shard_size="2GB"
        )
    
    # Save tokenizer in both cases
    tokenizer.save_pretrained(save_dir)
    
    # Upload to Hub
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, token=hf_token)
        api.upload_folder(
            folder_path=save_dir,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token
        )
        print(f"\nModel uploaded to: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"Upload error: {str(e)}")


def get_training_args() -> TrainingArguments:
    """Configure training arguments optimized for RTX 4090 under WSL2.
    
    WSL2 advantages over Windows:
    - Better process handling and memory management
    - Direct GPU access through CUDA
    - More efficient filesystem operations
    - Better multiprocessing support
    """
    return TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,                    # Total number of training epochs
        
        # Batch Size Configuration
        # WSL2 has better memory management than Windows, allowing for more aggressive batching
        # RTX 4090 has 24GB VRAM and WSL2 can utilize it more efficiently
        # Effective batch size = per_device_batch * gradient_accumulation = 8 * 4 = 32
        per_device_train_batch_size=8,         # Optimal for 24GB VRAM under WSL2
        per_device_eval_batch_size=8,          # Match training batch size
        gradient_accumulation_steps=4,         # Accumulate for larger effective batch
        
        # Data Loading Optimization
        # WSL2's Linux kernel provides better process management than Windows
        # Can use more CPU cores efficiently without system instability
        # 12 workers (75% of cores) is optimal for:
        # - Leaving resources for system processes
        # - Maximizing data loading throughput
        # - Avoiding memory contention
        dataloader_num_workers=12,             # WSL2 handles more workers efficiently
        dataloader_pin_memory=True,            # Fast GPU transfer via pinned memory
        
        # Training Schedule
        # WSL2's better I/O handling makes frequent evaluations less costly
        warmup_steps=100,                      # Standard warmup for stability
        evaluation_strategy="steps",           # Regular evaluation intervals
        eval_steps=100,                        # Evaluate every 100 steps
        save_strategy="steps",                 # Checkpoint intervals
        save_steps=100,                        # Save every 100 steps
        save_total_limit=3,                    # Manage disk space
        
        # Optimizer Configuration
        # WSL2's memory management allows for stable training with these settings
        learning_rate=2e-4,                    # Standard for LoRA fine-tuning
        weight_decay=0.01,                     # L2 regularization
        lr_scheduler_type="cosine",            # Smooth LR decay
        
        # GPU Optimization
        # WSL2 provides direct GPU access via CUDA, making these optimizations fully effective
        # No Windows overhead in the GPU access path
        fp16=True,                            # Mixed precision training
        tf32=True,                            # Tensor cores optimization
        
        # Logging Configuration
        # WSL2's filesystem is more efficient for frequent small writes
        logging_steps=10,                      # Frequent logging is less costly
        report_to="wandb",                     # Remote metric tracking
        
        # Training Optimizations
        # WSL2's better memory management makes these optimizations more effective
        group_by_length=True,                 # Efficient sequence batching
        length_column_name="length",          # Required for length grouping
        gradient_checkpointing=False,         # Not needed with 24GB VRAM
        
        # Model Selection
        # WSL2's efficient I/O makes model saving/loading faster
        load_best_model_at_end=True,          # Keep best checkpoint
        metric_for_best_model="eval_loss",    # Optimization target
        greater_is_better=False,              # Minimize loss
        
        # Training Stability
        # WSL2's consistent performance helps maintain stable training
        max_grad_norm=1.0,                    # Prevent gradient explosions
        
        # WSL2 Specific Settings
        # WSL2 uses the Linux kernel, so we can skip Windows-specific tweaks
        # - No need for Windows memory optimizations
        # - No need for Windows process handling adjustments
        # - Can use Linux-native CUDA performance
        use_mps_device=False,                 # WSL2 uses CUDA directly
    )


# For testing, let's use fewer prompts
if __name__ == "__main__":
    try:
        # Setup
        model, tokenizer = prepare_fine_tuning()
        train_dataset, eval_dataset = prepare_dataset(tokenizer)
        
        print(f"\nTraining with {len(train_dataset)} training samples and {len(eval_dataset)} eval samples")

        # Test prompts for callback
        test_prompts = [
            "your neighbor who plays loud music at 3am",
            "the customer service representative who hung up on you",
            "the restaurant that gave you food poisoning",
            "the delivery driver who left your package in the rain"
        ]

        trainer = Trainer(
            model=model,
            args=get_training_args(),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[
                ComplaintTestingCallback(test_prompts, tokenizer, every_n_steps=100),
                EarlyStoppingCallback(early_stopping_patience=3),
            ],
        )

        print("\nStarting training...")
        trainer.train()

        # Save model
        model_name = "unsloth/Llama-3.2-1B-Instruct"
        merged_name = "Llama-3.2-1B-Instruct-Complaint"
        repo_id = f"leonvanbokhorst/{merged_name}"

        save_model(
            model=model,
            tokenizer=tokenizer,
            save_dir=f"./complaint_model/{merged_name}",
            repo_id=repo_id,
            hf_token=HF_TOKEN,
            save_mode="merged"
        )

    finally:
        wandb.finish()
