import torch
from diffusers import StableDiffusionPipeline
import cv2
import numpy as np
from PIL import Image
import os

# Configura el modelo de Stable Diffusion (requiere GPU para velocidad)
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Usar GPU (o "cpu" si no tienes)

def text_to_video(prompt, output_video="output.mp4", fps=24, duration=5):
    # Generar frames (imágenes)
    frames = []
    num_frames = fps * duration
    
    for _ in range(num_frames):
        image = pipe(prompt).images[0]
        frames.append(np.array(image))
    
    # Guardar frames como video
    height, width, _ = frames[0].shape
    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    video.release()
    print(f"Video guardado como {output_video}")

# Ejemplo de uso
text_to_video("Un gato astronauta en Marte", "gato_astronauta.mp4")
