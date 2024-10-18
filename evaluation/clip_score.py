from tqdm import tqdm
from PIL import Image
import torch
import os
import numpy as np
from omegaconf import OmegaConf
from transformers import CLIPProcessor, CLIPModel
import argparse
parser = argparse.ArgumentParser(description="compute clip score")
parser.add_argument(
    "--target_folder",
    type=str,
)
parser.add_argument(
    "--prefix_name",
    default=None,
    type=str,
)
args = parser.parse_args()
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

model.cuda()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_clip_score(image_path, text):
    image = Image.open(image_path).resize((512,512))
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    return logits_per_image

def extract_id(filename):
    id_str = filename.split('-')[0]
    return int(id_str)


config = OmegaConf.load('large_scale_prompts.yaml')
prompts=config['prompt']
target_folder=args.target_folder
sim_imgs = None
sim_texts = None
files = os.listdir(target_folder)
files = sorted(files, key=extract_id)
with torch.no_grad():
    for idx,file in enumerate(tqdm(files, desc="number")):
        image_path = os.path.join(target_folder, file)
        prompt = prompts[idx]
        if args.prefix_name!=None:
            prompt=args.prefix_name+prompt
        prompt = [prompt]
        sim_text = get_clip_score(image_path, prompt)
        if sim_texts is None:
            sim_texts = sim_text
        else:
            sim_texts = torch.cat([sim_texts, sim_text])

print("Text similarity: %.6f"%(sim_texts.mean()))