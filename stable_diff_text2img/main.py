from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline
import torch
from datetime import datetime
import os

MODEL_ID = "leonvanbokhorst/Llama-3.2-1B-Instruct-Complaint"
SD_MODEL_ID = "CompVis/stable-diffusion-v1-4"  # of jouw specifieke SD-model

# Laad tokenizer en taalmodel
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="right", add_eos_token=True, add_bos_token=True,
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cpu")

PROMPT_TEMPLATE = """### Instruction:
{0}

### Response:
"""

# zorgen dat die een prompt terug geeft vvooor het aanmaken voor images hij schrijft nu hele raren prompts
def generate_text_prompt(prompt: str) -> str:
    instruction = f"Write an angry complaint about {prompt}."
    formatted_prompt = PROMPT_TEMPLATE.format(instruction)
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True).to("cpu")
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.9, top_p=0.9, do_sample=True,
                             repetition_penalty=1.2)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("### Response:")[-1].strip()

# aanpassen zorgen dat die met je proccesor werkt"
# Laad Stable Diffusion pipeline
sd_pipe = StableDiffusionPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=torch.float16).to("cpu")

# Zorg ervoor dat de outputmap bestaat
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# de code waar je mee moet werken
def generate_image_from_text(text_prompt: str):
    # Genereer de afbeelding
    image = sd_pipe(text_prompt, width=1280, height=720).images[0]# gebruik hier de model van leo

    # Genereer een unieke bestandsnaam met datum, tijd en prompt
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_prompt = "_".join(text_prompt.split())[:50]  # Verkort en vervang spaties door underscores
    filename = f"{timestamp}_{sanitized_prompt}.png"

    # Sla de afbeelding op
    image_path = os.path.join(output_dir, filename)
    image.save(image_path)

    print(f"Afbeelding opgeslagen als: {image_path}")


# Test met een prompt
text_prompt = generate_text_prompt("What is the capital of the Netherlands?")
print("Generated Text Prompt:", text_prompt)

generate_image_from_text("generate a apple") # hier moet je straks de teskt van de prompt gebruiken maar nu kan je gewoon strings erin zetten.

