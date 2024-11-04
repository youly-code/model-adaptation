import torch
from diffusers import StableDiffusionPipeline

# Kies het model dat je wilt gebruiken
MODEL_ID = "CompVis/stable-diffusion-v1-4"
device = "cpu"  # Houd de CPU aan als je geen GPU hebt

print(f"🖼️ Loading model: {MODEL_ID}")

# Laad de beeldgeneratiepipeline en wijzig torch_dtype naar float32 voor CPU
pipeline = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
pipeline.to(device)

print("✓ Model loaded and ready to generate images")

def generate_image(prompt: str):
    print(f"Generating image for prompt: {prompt}")
    image = pipeline(prompt).images[0]  # Genereer het beeld
    image.show()  # Toon het beeld
    return image

# Test het model met een voorbeeldprompt
prompt = "a cozy cabin in the snowy mountains at sunset"
generated_image = generate_image(prompt)

image = pipeline(prompt, num_inference_steps=25, height=256, width=256).images[0]
