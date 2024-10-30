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
)
from peft import LoraConfig, get_peft_model
import os
import dotenv
import torch

dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


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
    """Setup and prepare model for LoRA fine-tuning
    
    Returns:
        tuple: (model, tokenizer) pair configured for training
        
    Raises:
        ValueError: If HF_TOKEN is not set
        RuntimeError: If CUDA/MPS device initialization fails
    """
    model_name = "unsloth/Llama-3.2-1B"

    if not HF_TOKEN:
        raise ValueError("Please set the HF_TOKEN environment variable")

    try:
        # Determine optimal device
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            token=HF_TOKEN,
            device_map=device,
            torch_dtype=torch.float16
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model: {str(e)}")

    tokenizer = initialize_tokenizer(model_name, HF_TOKEN)

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    return model, tokenizer


def prepare_dataset(tokenizer):
    """Prepare dataset for fine-tuning"""
    try:
        dataset = load_dataset("leonvanbokhorst/synthetic-complaints")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {str(e)}")

    print("Dataset structure:", dataset["train"].features)
    print("First example:", dataset["train"][0])

    def format_prompt(example):
        """Format each example into Llama instruction format"""
        return {
            "text": f"[INST] {example['instruction']} [/INST] {example['response']}"
        }

    def tokenize_function(examples):
        """Tokenize the formatted examples
        
        Args:
            examples: Batch of examples to tokenize
            
        Returns:
            dict: Tokenized examples
        """
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )

    print("Formatting prompts...")
    formatted_dataset = dataset["train"].map(format_prompt)

    print("Tokenizing dataset...")
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing",
        remove_columns=formatted_dataset.column_names,
    )

    return tokenized_dataset


def train_model(model, tokenizer, dataset):
    """Train the model using standard training
    
    Training parameters optimized for M3 architecture:
    - batch_size=2: Balanced for memory constraints
    - gradient_accumulation=4: Effective batch size of 8
    - learning_rate=2e-4: Empirically optimal for LoRA
    - weight_decay=0.1: Prevents overfitting on small datasets
    
    Args:
        model: Model to train
        tokenizer: Tokenizer instance
        dataset: Processed dataset
        
    Returns:
        Trainer: Trained model trainer instance
    """
    training_args = TrainingArguments(
        output_dir="./lora_finetuned",           # Directory where model checkpoints will be saved
        num_train_epochs=1,                      # Number of complete passes through the dataset
        per_device_train_batch_size=2,           # Number of samples processed on each device per batch
        gradient_accumulation_steps=4,           # Number of batches to accumulate before performing a backward/update pass
        save_steps=100,                          # Save checkpoint every X steps
        logging_steps=25,                        # Log training metrics every X steps
        learning_rate=2e-4,                      # Initial learning rate for training
        weight_decay=0.1,                        # L2 regularization factor to prevent overfitting
        fp16=False,                              # Whether to use 16-bit floating point precision
        optim="adamw_torch",                     # Optimizer type (AdamW is standard for transformer models)
        lr_scheduler_type="cosine_with_restarts" # Learning rate schedule - gradually decreases LR with periodic restarts
    )

    # Initialize trainer with progress bar
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting training...")
    trainer.train()

    return trainer


def inference_example(model, tokenizer, prompt: str) -> str:
    """Generate text using fine-tuned model
    
    Args:
        model: Fine-tuned model
        tokenizer: Associated tokenizer
        prompt: Input prompt text
        
    Returns:
        str: Generated response text
        
    Raises:
        RuntimeError: If generation fails
    """
    try:
        formatted_prompt = f"[INST] {prompt} [/INST]"
        device = model.device
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            do_sample=True,
            num_return_sequences=1,
        )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        raise RuntimeError(f"Generation failed: {str(e)}")


# Usage example
if __name__ == "__main__":
    FINETUNING = True

    if FINETUNING:
        # Setup
        model, tokenizer = prepare_fine_tuning()

        # Prepare dataset
        dataset = prepare_dataset(tokenizer)

        # Train
        trainer = train_model(model, tokenizer, dataset)

        # Save
        model.save_pretrained("./lora_finetuned")
        tokenizer.save_pretrained("./lora_finetuned")

        # Example inference after training
        prompt = "Write a frustrated complaint about poor customer service"
        response = inference_example(model, tokenizer, prompt)
        print(response)

    else:
        # Load
        tokenizer = AutoTokenizer.from_pretrained("./lora_finetuned")
        model = AutoModelForCausalLM.from_pretrained("./lora_finetuned")

        # Example inference
        prompt = "Instruction: Generate a lighthearted joke about AI ethics and bias in a Professional setting"
        response = inference_example(model, tokenizer, prompt)
        print(response)
