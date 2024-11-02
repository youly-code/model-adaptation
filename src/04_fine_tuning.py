from tqdm import tqdm
from datasets import load_dataset
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
from dataclasses import dataclass
from typing import List, Dict, Any
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
        bnb_4bit_quant_storage=torch.bfloat16,  # Added for better precision
    )

    # Initialize model with stricter memory constraints
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
        torch_dtype=torch.float16,
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
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
        lora_dropout=0.15,
        bias="none",
        task_type="CAUSAL_LM",
        fan_in_fan_out=False,
        modules_to_save=[
            "embed_tokens",
            "lm_head",
        ],  # This is sufficient to handle embeddings
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
    if special_char_ratio > 0.2:  # Increased from 0.1
        return False

    return True


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
        instruction = f"Tell me about {example['topic']}"
        output = example["output"]

        return {"instruction": instruction, "output": output}

    train_dataset = split_dataset["train"].map(prepare_prompt)
    eval_dataset = split_dataset["test"].map(prepare_prompt)

    # Print first example before tokenization
    first_example = train_dataset[0]
    formatted_prompt = alpaca_prompt.format(
        first_example["instruction"], first_example["output"]
    )
    print("\nFirst training prompt:")
    print(formatted_prompt)

    def tokenize_function(examples):
        """Tokenize using simplified template with proper labels"""
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

        # Mask labels before the response section (we only want to train on the response)
        for idx, text in enumerate(texts):
            # Find the position of "### Response:" in the tokenized input
            response_token_ids = tokenizer.encode("### Response:")
            input_ids = tokenized["input_ids"][idx].tolist()

            # Find the start of the response section
            for i in range(len(input_ids) - len(response_token_ids)):
                if input_ids[i : i + len(response_token_ids)] == response_token_ids:
                    # Mask everything before the response with -100
                    tokenized["labels"][idx, : i + len(response_token_ids)] = -100
                    break

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


def train_model(model, tokenizer, train_dataset, eval_dataset):
    """Training with proper configuration for stable generation"""
    training_args = TrainingArguments(
        output_dir="./complaint_model",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
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
        warmup_ratio=0.5,
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
        callbacks=[
            ComplaintTestingCallback(test_prompts, tokenizer, every_n_steps=100)
        ],
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
                loss = loss["loss"]
            elif hasattr(loss, "loss"):
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


def save_lora_adapter(
    model, tokenizer, save_dir: str, repo_id: str, hf_token: str
) -> None:
    """Save LoRA adapter locally and upload to Hugging Face Hub.

    Args:
        model: The trained PEFT model
        tokenizer: The tokenizer used for training
        save_dir: Local directory to save adapter files
        repo_id: Hugging Face Hub repository ID (username/repo_name)
        hf_token: Hugging Face API token

    The LoRA (Low-Rank Adaptation) adapter contains:
    1. LoRA A matrices: Low-dimensional projection matrices (hidden_dim → rank)
    2. LoRA B matrices: Low-dimensional reconstruction matrices (rank → hidden_dim)
    3. Scaling factors: Applied during inference
    """
    os.makedirs(save_dir, exist_ok=True)

    # Get only the LoRA parameters with detailed filtering
    lora_state_dict = {
        k: v.to("cpu")
        for k, v in model.state_dict().items()
        if any(
            substr in k
            for substr in [
                "lora_A",  # Low-rank projection matrices
                "lora_B",  # Low-rank reconstruction matrices
                "lora_embedding_A",  # For embedding layers if used
                "lora_embedding_B",  # For embedding layers if used
                "lora_scaling",  # Scaling factors
            ]
        )
    }

    # Print detailed information about saved parameters
    print(f"\nLoRA Adapter Statistics:")
    print(f"Number of LoRA parameters: {len(lora_state_dict)}")
    print(f"Parameter names (first 5): {list(lora_state_dict.keys())[:5]}")

    total_params = sum(v.numel() for v in lora_state_dict.values())
    total_size_mb = sum(
        v.nelement() * v.element_size() for v in lora_state_dict.values()
    ) / (1024 * 1024)
    print(f"Total parameters: {total_params:,}")
    print(f"Adapter size: {total_size_mb:.2f}MB")

    # Save only the LoRA state dict
    adapter_path = os.path.join(save_dir, "adapter_model.bin")
    torch.save(lora_state_dict, adapter_path)
    print(f"\nAdapter saved to: {adapter_path}")

    # Save the configs
    if hasattr(model, "config"):
        model.config.save_pretrained(save_dir)

    if hasattr(model, "peft_config"):
        model.peft_config["default"].save_pretrained(save_dir)

    # Save tokenizer configuration
    tokenizer.save_pretrained(save_dir)

    # Create repository and upload
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, token=hf_token)

        # Upload adapter files
        api.upload_folder(
            folder_path=save_dir, repo_id=repo_id, repo_type="model", token=hf_token
        )

        print(f"\nLoRA adapter uploaded to: https://huggingface.co/{repo_id}")
        print("\nAdapter can be loaded with:")
        print(
            f"""
from peft import PeftModel, PeftConfig
config = PeftConfig.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, "{repo_id}")
        """
        )

    except Exception as e:
        print(f"Upload error: {str(e)}")


def save_merged_model(
    model, tokenizer, save_dir: str, repo_id: str, hf_token: str
) -> None:
    """Save the merged model (base + LoRA) and upload to Hugging Face Hub."""
    os.makedirs(save_dir, exist_ok=True)

    print("\nPreparing model for merge...")
    # Get the base model name from the PEFT config
    base_model_name = model.peft_config["default"].base_model_name_or_path

    # Load the base model in FP16 instead of 4-bit
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map="auto", token=hf_token
    )

    print("\nMerging LoRA adapter with base model...")
    # Merge LoRA weights with the FP16 base model
    model = PeftModel.from_pretrained(base_model, model.peft_config["default"])
    merged_model = model.merge_and_unload()

    print(f"\nSaving merged model to: {save_dir}")
    merged_model.save_pretrained(
        save_dir, safe_serialization=True, max_shard_size="2GB"
    )
    tokenizer.save_pretrained(save_dir)

    # Create repository and upload
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, token=hf_token)

        print(f"\nUploading merged model to: https://huggingface.co/{repo_id}")
        api.upload_folder(
            folder_path=save_dir, repo_id=repo_id, repo_type="model", token=hf_token
        )

        print("\nMerged model can be loaded with:")
        print(
            f"""
model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
        """
        )

    except Exception as e:
        print(f"Upload error: {str(e)}")


# For testing, let's use fewer prompts
if __name__ == "__main__":
    FINETUNING = True
    TESTING = False

    if FINETUNING:
        try:
            # Setup
            model, tokenizer = prepare_fine_tuning()
            train_dataset, eval_dataset = prepare_dataset(tokenizer)

            # Limit dataset sizes for testing
            if TESTING:
                train_dataset = train_dataset.select(range(1000))
                eval_dataset = eval_dataset.select(range(100))

            print(
                f"\nTraining with {len(train_dataset)} training samples and {len(eval_dataset)} eval samples"
            )

            # Create a single function for training arguments
            def get_training_args(run_name: str = None) -> TrainingArguments:
                """Create standardized training arguments"""
                return TrainingArguments(
                    output_dir="./complaint_model",
                    run_name=run_name
                    or f"complaint-training-{wandb.util.generate_id()}",
                    num_train_epochs=2,
                    per_device_train_batch_size=16,
                    per_device_eval_batch_size=1,
                    gradient_accumulation_steps=8,
                    learning_rate=1e-4,
                    warmup_ratio=0.5,
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
                    seed=42,
                    dataloader_drop_last=True,
                )

            # Simple test prompts
            test_prompts = [
                "How's the weather?",
                "Tell me about your neighbor's habits",
                "What's your opinion on modern workplace culture?",
                "What's your favorite book?",
                "Tell me about your favorite vacation spot",
            ]

            trainer = CustomTrainer(
                model=model,
                args=get_training_args(),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                callbacks=[
                    ComplaintTestingCallback(
                        test_prompts, tokenizer, every_n_steps=100
                    ),
                    EarlyStoppingCallback(early_stopping_patience=3),
                ],
            )

            print("\nStarting training... This may take a while. 😁")
            trainer.train()

            # After training, merge the LoRA weights and save the full model
            print("\nMerging LoRA weights and saving full model...")
            # First merge the LoRA weights
            model = model.merge_and_unload()

            # Save and reload the model in float16 instead of direct conversion
            temp_path = "./complaint_model/temp_merged"
            model.save_pretrained(temp_path)
            
            model = AutoModelForCausalLM.from_pretrained(
                temp_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            # Continue with Hub upload
            model_name = "unsloth/Llama-3.2-1B-Instruct"
            merged_name = "Llama-3.2-1B-Instruct-Complaint"
            repo_id = f"leonvanbokhorst/{merged_name}"

            # Save the full merged model
            model.save_pretrained(
                f"./complaint_model/{merged_name}",
                push_to_hub=True,
                use_auth_token=HF_TOKEN,
                repo_id=repo_id,
                safe_serialization=True,
            )

            # Save tokenizer
            tokenizer.save_pretrained(
                f"./complaint_model/{merged_name}",
                push_to_hub=True,
                use_auth_token=HF_TOKEN,
                repo_id=repo_id,
            )

            print(f"\nMerged model saved to: https://huggingface.co/{repo_id}")

        finally:
            wandb.finish()
