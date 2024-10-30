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

# Add at the top of the file, after imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    model_name = "unsloth/Llama-3.2-1B"

    model_kwargs = {
        "token": HF_TOKEN,
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }

    try:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Load model without device_map first
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
            use_cache=False,
        )

        # Move model to device after loading
        model = model.to(device)

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    except Exception as e:
        raise RuntimeError(f"Failed to initialize model: {str(e)}") from e

    tokenizer = initialize_tokenizer(model_name, HF_TOKEN)

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
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
        """Tokenize the formatted examples"""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,  # Increased back to 512
            padding="max_length",
            return_tensors="pt",
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
    """Train the model using optimized training parameters

    Key training parameters:
    - batch_size=2 with gradient_accumulation=4: Provides effective batch size of 8
      while staying within memory constraints
    - learning_rate=2e-4: Empirically optimal for LoRA fine-tuning
    - weight_decay=0.01: Prevents overfitting while allowing adaptation
    - gradient_checkpointing=True: Reduces memory usage during training
    - warmup_ratio=0.1: Helps stabilize early training
    """
    training_args = TrainingArguments(
        output_dir="./lora_finetuned",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        save_steps=100,
        logging_steps=20,
        learning_rate=2e-4,
        weight_decay=0.01,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        group_by_length=True,
        report_to="none",
        save_total_limit=2,
    )

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
    """Generate text using fine-tuned model"""
    try:
        formatted_prompt = f"[INST] {prompt} [/INST]"
        device = model.device
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {
            k: v.to(device) for k, v in inputs.items()
        }  # Ensure inputs are on same device as model

        outputs = model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            do_sample=True,
            num_return_sequences=1,
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        raise RuntimeError(f"Generation failed: {str(e)}") from e


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
    else:
        # Load
        tokenizer = AutoTokenizer.from_pretrained("./lora_finetuned")
        model = AutoModelForCausalLM.from_pretrained("./lora_finetuned")

        # Example inference
        prompt = "Instruction: Generate a lighthearted joke about AI ethics and bias in a Professional setting"

    response = inference_example(model, tokenizer, prompt)
    print(response)
