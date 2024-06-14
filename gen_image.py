import torch
from diffusers import StableDiffusionPipeline

class GenImage:
    def __init__(self, model_name="CompVis/stable-diffusion-v1-4"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = StableDiffusionPipeline.from_pretrained(self.model_name).to(self.device)

    def do(self, prompt):
        image = self.pipeline(prompt).images[0]
        output_file = "output_image.png"
        image.save(output_file)
        print(f"Generated image saved as {output_file}")
        return output_file