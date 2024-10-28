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


def prepare_fine_tuning():
    """Setup and prepare model for LoRA fine-tuning"""

    # Update the model name to the correct format
    model_name = "unsloth/Llama-3.2-1B"
    hf_token = HF_TOKEN

    if not hf_token:
        raise ValueError(
            "Please set the HF_TOKEN environment variable with your Hugging Face token. "
            "You can get it from https://huggingface.co/settings/tokens"
        )

    # Use AutoTokenizer instead of LlamaTokenizer
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

    # Use LlamaForCausalLM directly
    model = LlamaForCausalLM.from_pretrained(
        model_name, device_map="auto", token=hf_token
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # Rank of update matrices
        lora_alpha=32,  # Alpha scaling factor
        target_modules=["q_proj", "v_proj"],  # Which modules to apply LoRA to
        lora_dropout=0.05,  # Dropout probability
        bias="none",  # Don't train bias parameters
        task_type="CAUSAL_LM",  # Task type for causality
    )

    # Create PEFT model
    model = get_peft_model(model, lora_config)

    return model, tokenizer


def prepare_dataset(tokenizer):
    """Prepare dataset for fine-tuning"""

    print("Loading dataset...")
    dataset = load_dataset("csv", data_files="src/data/ai_ethics_jokes_training.csv")

    def format_prompt(example):
        """Format each example into instruction-following format"""
        return {
            "text": f"Instruction: {example['instruction']}\n"
            f"Input: {example['input']}\n"
            f"Output: {example['output']}\n"
            f"Context: {example['context']}\n"
            f"Type: {example['type']}\n"
            f"Difficulty: {example['difficulty']}"
        }

    print("Formatting prompts...")
    formatted_dataset = dataset.map(
        format_prompt,
        desc="Formatting prompts"  # Removed the disable parameter
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        )

    print("Tokenizing dataset...")
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing",
        remove_columns=formatted_dataset["train"].column_names,
    )

    return tokenized_dataset


def train_model(model, tokenizer, dataset):
    """Train the model using LoRA"""

    # Check if GPU is available
    use_fp16 = torch.cuda.is_available()  # Only use fp16 if CUDA GPU is available

    training_args = TrainingArguments(
        output_dir="./lora_finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_steps=100,
        logging_steps=100,
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=use_fp16,  # Dynamically set based on GPU availability
        push_to_hub=False,
    )

    # Initialize trainer with progress bar
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Train with progress bar
    print("Starting training...")
    trainer.train()

    return trainer


def inference_example(model, tokenizer, prompt):
    """Generate text using fine-tuned model"""

    # Ensure inputs are on the correct device
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Add attention mask if not present
    if 'attention_mask' not in inputs:
        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])

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
    # Setup
    model, tokenizer = prepare_fine_tuning()

    # Prepare dataset
    dataset = prepare_dataset(tokenizer)

    # Train
    trainer = train_model(model, tokenizer, dataset)

    # Save
    model.save_pretrained("./lora_finetuned_model")

    # Example inference
    prompt = "Instruction: Generate a lighthearted joke about AI ethics\nInput: Topic: AI bias\nContext: Professional setting\nType: Humor\nDifficulty: Medium"
    response = inference_example(model, tokenizer, prompt)
    print(response)
