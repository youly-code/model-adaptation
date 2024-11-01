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
import bitsandbytes as bnb

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
        
    if tokenizer.chat_template is None:
        tokenizer.chat_template = """{% for message in messages %}
{% if message['role'] == 'system' %}{% if not loop.first %}
{% endif %}{{ message['content'] }}
{% elif message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}
<|im_end|>
{% elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{ message['content'] }}
<|im_end|>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<|im_start|>assistant
{% endif %}"""

    return tokenizer


def prepare_fine_tuning():
    """Setup for lower memory usage"""
    model_name = "unsloth/Llama-3.2-1B"
    
    # More aggressive QLoRA configuration
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
        torch_dtype=torch.float16,
    )
    
    # Initialize tokenizer with chat template
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add ChatML-style template
    tokenizer.chat_template = """{% for message in messages %}
{% if message['role'] == 'system' %}{% if not loop.first %}
{% endif %}{{ message['content'] }}
{% elif message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}
<|im_end|>
{% elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{ message['content'] }}
<|im_end|>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<|im_start|>assistant
{% endif %}"""

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
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
        example: Dataset example containing 'output' text
        
    Returns:
        bool: True if example meets quality criteria
    """
    text = example['output']
    
    # Skip empty or very short responses
    if not text or len(text.split()) < 10:
        return False
        
    # Skip responses with excessive special characters
    special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,!?]', text)) / len(text)
    if special_char_ratio > 0.1:
        return False
        
    # Skip responses with repetitive patterns
    if any(text.count(phrase) > 2 for phrase in text.split() if len(phrase) > 3):
        return False
        
    return True


def prepare_dataset(tokenizer):
    """Prepare dataset with standard chat template"""
    dataset = load_dataset("leonvanbokhorst/synthetic-complaints-v2")
    
    # Reduce validation set size
    dataset = dataset["train"].select(range(30000))
    filtered_dataset = dataset.filter(filter_quality)
    split_dataset = filtered_dataset.train_test_split(test_size=0.05, seed=42)
    
    def prepare_prompt(example):
        """Create consistent prompt structure using chat format"""
        messages = [
            {
                "role": "system", 
                "content": "You are an expert at writing detailed complaints and criticisms. Your responses should be negative but realistic, focusing on specific issues and their impact."
            },
            {"role": "user", "content": f"Tell me about {example['topic']}"},
            {"role": "assistant", "content": example['output']}
        ]
        return {"messages": messages}
    
    train_dataset = split_dataset["train"].map(prepare_prompt)
    eval_dataset = split_dataset["test"].map(prepare_prompt)

    def tokenize_function(examples):
        """Tokenize using chat template"""
        # Apply chat template to each example
        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            for messages in examples["messages"]
        ]
        
        # Tokenize the formatted texts
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        tokenized["labels"] = tokenized["input_ids"].clone()
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
    """Memory-optimized negativity score calculation"""
    # Use only VADER for sentiment analysis (more memory efficient)
    # try:
    #     nltk.data.find("sentiment/vader_lexicon.zip")
    # except LookupError:
    #     nltk.download("vader_lexicon", quiet=True)

    # sia = SentimentIntensityAnalyzer()
    # vader_scores = sia.polarity_scores(text)

    # # Convert compound score to 0-1 range where 1 is most negative
    # return (vader_scores["compound"] + 1) / 2
    pass


def compute_metrics(eval_preds) -> Dict[str, float]:
    # """Compute custom metrics for complaint quality"""
    # predictions, labels = eval_preds
    # if isinstance(predictions, tuple):
    #     predictions = predictions[0]

    # # Ensure we're working with numpy arrays
    # predictions = predictions.astype(np.int64)

    # try:
    #     # Process in batches
    #     decoded_preds = []
    #     for pred in predictions:
    #         try:
    #             # Create boolean masks for filtering
    #             not_pad_mask = pred != tokenizer.pad_token_id
    #             not_ignore_mask = pred != -100
    #             valid_tokens_mask = not_pad_mask & not_ignore_mask

    #             # Filter tokens using the mask
    #             valid_tokens = pred[valid_tokens_mask].tolist()

    #             # Decode filtered tokens
    #             text = tokenizer.decode(valid_tokens, skip_special_tokens=True)
    #             decoded_preds.append(text)
    #         except Exception as e:
    #             print(f"Decoding error: {e}")
    #             decoded_preds.append("")

    #     # Calculate metrics
    #     negativity_scores = []
    #     for text in decoded_preds:
    #         if text.strip():  # Only process non-empty strings
    #             try:
    #                 score = calculate_negativity(text)
    #                 if isinstance(score, (float, int)):  # Ensure score is a scalar
    #                     negativity_scores.append(score)
    #             except Exception as e:
    #                 print(f"Scoring error: {e}")

    #     # Return average scores, defaulting to 0 if no valid scores
    #     avg_negativity = float(np.mean(negativity_scores)) if negativity_scores else 0.0
    #     return {"negativity": avg_negativity}

    # except Exception as e:
    #     print(f"Metrics computation error: {e}")
    return {"negativity": 0.5}


def train_model(model, tokenizer, train_dataset, eval_dataset):
    """Training with proper configuration for stable generation"""
    training_args = TrainingArguments(
        output_dir="./complaint_model",
        num_train_epochs=2,
        # Smaller batches for stability
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=16,
        # Conservative learning settings
        learning_rate=5e-5,
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        weight_decay=0.02,
        # Evaluation settings
        eval_steps=500,  # Increased to reduce validation frequency
        max_steps=2000,
        evaluation_strategy="steps",
        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
    """Generate responses with fixed configuration"""
    try:
        device = model.device
        messages = [
            {
                "role": "system",
                "content": "You are an expert at writing detailed complaints and criticisms. Write a specific, focused complaint about the given topic. Focus on concrete issues, their impacts, and maintain a professional tone."
            },
            {"role": "user", "content": f"Write a detailed complaint about {prompt}, focusing on specific issues and their negative impacts."}
        ]
        
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer(
            chat_text,
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
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                early_stopping=True,
                num_beams=4,
                repetition_penalty=1.3,
                no_repeat_ngram_size=4,
                eos_token_id=tokenizer.eos_token_id,
                length_penalty=1.0,
            )

        # Improved response extraction
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # More robust pattern matching
        assistant_pattern = r'<\|im_start\|>assistant\n(.*?)(?:<\|im_end\||$)'
        assistant_response = re.search(assistant_pattern, response, re.DOTALL)
        
        if assistant_response:
            response = assistant_response.group(1).strip()
            # Clean up any remaining artifacts
            response = re.sub(r'<\|.*?\|>', '', response).strip()
            return response
        return "Failed to extract valid response"

    except Exception as e:
        print(f"Generation error: {str(e)}")
        return f"Generation failed: {str(e)}"


class CustomTrainer(Trainer):
    """Custom trainer with proper gradient handling"""

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step with proper gradient scaling"""
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            if self.args.fp16 or self.args.bf16:
                with torch.amp.autocast('cuda'):  # Updated autocast usage
                    loss = self.compute_loss(model, inputs)
                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps
                
                self.accelerator.backward(loss)
            else:
                loss = self.compute_loss(model, inputs)
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                
                loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
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

            # Training arguments optimized for 24GB GPU
            training_args = TrainingArguments(
                output_dir="./complaint_model",
                run_name=f"complaint-training-{wandb.util.generate_id()}",
                num_train_epochs=2,
                per_device_train_batch_size=24,
                per_device_eval_batch_size=24,
                gradient_accumulation_steps=6, # size recommended by HF for 24GB GPU = 1/4 batch size which is 6
                max_grad_norm=0.5,
                bf16=False,
                fp16=True,
                learning_rate=2e-5,
                logging_steps=10,
                evaluation_strategy="epoch",
                eval_steps=1,
                save_strategy="epoch",
                save_steps=1,
                weight_decay=0.01,
                report_to="wandb",
                load_best_model_at_end=True,
                optim="adamw_torch",
                warmup_ratio=0.03,
                lr_scheduler_type="cosine",
                group_by_length=True,
                gradient_checkpointing=True,
                fp16_full_eval=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                dataloader_num_workers=2,
                dataloader_pin_memory=True,
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
                    ComplaintTestingCallback(test_prompts, tokenizer, every_n_steps=100)
                ],
            )

            print("\nStarting training... This may take a while. 😁")
            trainer.train()

            # print("\nTesting inference: ")
            # for test_prompt in test_prompts:
            #     response = inference_example(model, tokenizer, test_prompt)
            #     print(f"Prompt: {test_prompt}")
            #     print(f"Response: {response}")

        finally:
            wandb.finish()
