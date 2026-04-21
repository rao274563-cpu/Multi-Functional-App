from diffusers import StableDiffusionPipeline
import torch

class ImageGenerator:
    def __init__(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
        )
        self.pipe = self.pipe.to("cpu")

    def generate(self, prompt):
        image = self.pipe(prompt).images[0]
        image.save("generated_image.png")
        return "Image saves as generated_image.png"    