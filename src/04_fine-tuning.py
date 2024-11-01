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
import random

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

    # Configure LoRA with slightly adjusted parameters for better adaptation
    lora_config = LoraConfig(
        r=16,  # Increased from 8 to allow more expressiveness
        lora_alpha=32,  # Increased from 16 to strengthen adaptation
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Added k_proj and o_proj
        lora_dropout=0.1,  # Slightly increased dropout
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    return model, tokenizer


def prepare_dataset(tokenizer):
    """Prepare dataset for fine-tuning with focus on negative complaints"""
    try:
        dataset = load_dataset("leonvanbokhorst/synthetic-complaints-v2")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {str(e)}")

    # Filter for more negative complaints
    def filter_complaints(example):
        return (
            example['sentiment'] < 0.3  # Even more negative sentiment
            and example['subjectivity'] > 0.6  # More subjective complaints
            and example['complexity_score'] > 0.7  # More sophisticated complaints
        )

    # Take a larger subset but still manageable for testing
    filtered_dataset = dataset["train"].filter(filter_complaints)
    test_dataset = filtered_dataset.select(range(min(500, len(filtered_dataset))))  # Increased to 500 examples
    print(f"Test dataset size: {len(test_dataset)} (from {len(dataset['train'])})")

    def format_prompt(example):
        """Format each example into Llama instruction format without explicit complaint instructions"""
        return {
            "text": f"[INST] Tell me about {example['topic']} [/INST] {example['output']}"
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
    formatted_dataset = test_dataset.map(format_prompt)

    print("Tokenizing dataset...")
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing",
        remove_columns=formatted_dataset.column_names,
    )

    return tokenized_dataset


def train_model(model, tokenizer, dataset):
    """Train the model with enhanced parameters"""
    training_args = TrainingArguments(
        output_dir="./complaint_model_enhanced",
        num_train_epochs=3,  # Increased to 3 epochs
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        save_steps=50,
        logging_steps=10,
        learning_rate=5e-4,  # Slightly increased
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
    """Generate text using fine-tuned complaint model"""
    try:
        formatted_prompt = f"[INST] Tell me about {prompt} [/INST]"
        device = model.device
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # More conservative generation parameters
        outputs = model.generate(
            **inputs,
            max_length=256,  # Reduced from 512
            min_length=20,   # Add minimum length
            temperature=0.7,  # Reduced from 0.9
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_time=10.0,   # Add timeout
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Generation failed: {str(e)}"  # Return error message instead of raising


# For testing, let's use fewer prompts
if __name__ == "__main__":
    FINETUNING = True

    if FINETUNING:
        # Setup and training
        model, tokenizer = prepare_fine_tuning()
        dataset = prepare_dataset(tokenizer)
        trainer = train_model(model, tokenizer, dataset)
        
        # Save the model
        model.save_pretrained("./complaint_model_enhanced")
        tokenizer.save_pretrained("./complaint_model_enhanced")

        # More diverse test prompts
        test_prompts = [
            "your morning coffee",
            "social media",
            "the weather",
            "public transportation",
            "modern smartphones",
            "streaming services",
            "working from home",
            "grocery shopping"
        ]
        
        print("\nTesting generation with various prompts:")
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            try:
                response = inference_example(model, tokenizer, prompt)
                print(f"Response: {response}")
            except Exception as e:
                print(f"Error during generation: {str(e)}")
