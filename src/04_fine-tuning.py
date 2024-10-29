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
    """Setup and prepare model for LoRA fine-tuning"""
    model_name = "unsloth/Llama-3.2-1B"  # Or another compatible model

    if not HF_TOKEN:
        raise ValueError("Please set the HF_TOKEN environment variable")

    # Standard initialization instead of Unsloth
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=HF_TOKEN, device_map="auto", torch_dtype=torch.float16
    )

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

    def tokenize_function(examples):
        """Tokenize the text examples"""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors=None,
        )

    print("Loading dataset...")
    dataset = load_dataset("csv", data_files="src/data/ai_ethics_jokes_training.csv")

    # Keep only necessary columns
    dataset = dataset["train"].remove_columns(
        ["timestamp", "difficulty", "type", "context"]
    )

    def format_prompt(example):
        """Format each example into Llama instruction format"""
        return {
            "text": f"[INST] {example['instruction']}\n{example['input']} [/INST] {example['output']}"
        }

    print("Formatting prompts...")
    formatted_dataset = dataset.map(format_prompt)

    print("Tokenizing dataset...")
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing",
        remove_columns=formatted_dataset.column_names,
    )

    return tokenized_dataset


def train_model(model, tokenizer, dataset):
    """Train the model using standard training"""
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


def inference_example(model, tokenizer, prompt):
    """Generate text using fine-tuned model"""
    # Convert model to inference mode
    model = LanguageModel.for_inference(model)

    # Ensure inputs are on the correct device
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Add attention mask if not present
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    outputs = model.generate(
        **inputs,
        max_length=100,
        temperature=0.7,
        do_sample=True,  # Enable sampling for temperature to take effect
        num_return_sequences=1,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


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

    else:
        # Load
        tokenizer = AutoTokenizer.from_pretrained("./lora_finetuned")
        model = AutoModelForCausalLM.from_pretrained("./lora_finetuned")

        # Example inference
        prompt = "Instruction: Generate a lighthearted joke about AI ethics and bias in a Professional setting"
        response = inference_example(model, tokenizer, prompt)
        print(response)
