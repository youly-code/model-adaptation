import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "leonvanbokhorst/Llama-3.2-1B-Instruct-Complaint"

print(f"🤖 Loading model: {MODEL_ID}")

# Initialize tokenizer with exact same settings as training
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    padding_side="right",
    add_eos_token=True,
    add_bos_token=True,
    trust_remote_code=True  # Add this to ensure we load custom tokenizer settings
)

print("✓ Tokenizer initialized")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load the model with exact same settings as training
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map=None,
    trust_remote_code=True,  # Add this to ensure we load custom model settings
    use_safetensors=True
).to(device)

# Add this to your test.py to verify the model config
print("Model config:", model.config)
print("Model type:", type(model))

# Add this before the generate_complaint function
PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 23 July 2024

You are a helpful assistant<|eot_id|><|start_header_id|>user <|end_header_id|>

{0}<|eot_id|><|start_header_id|>assistant <|end_header_id|>
"""

def generate_complaint(prompt: str) -> str:
    # Make the instruction more explicit
    instruction = f"Tell me about {prompt}."
    
    formatted_prompt = PROMPT_TEMPLATE.format(instruction)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True).to(device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.9,  # Increase temperature for more expressive output
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    if "### Response:" in result:
        result = result.split("### Response:")[-1].strip()
    
    return result

# Test with a more complaint-inducing prompt
test_prompts = [
    "your neighbor who plays loud music at 3am",
    "the customer service representative who hung up on you",
    "the restaurant that gave you food poisoning",
    "the delivery driver who left your package in the rain"
]

for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    print(f"Response: {generate_complaint(prompt)}")
