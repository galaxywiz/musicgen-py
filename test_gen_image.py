import torch
from diffusers import StableDiffusionPipeline
import requests
from PIL import Image

# Load the Stable Diffusion pipeline
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

# Define your fantasy game-style prompt
prompt = "A majestic castle in a mystical forest, with dragons flying in the sky, in the style of a fantasy game concept art"

# Generate the image
with torch.no_grad():
    image = pipeline(prompt).images[0]

# Save the generated image
image.save("fantasy_game_art.png")

# Display the image
image.show()