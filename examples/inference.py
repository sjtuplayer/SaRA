import sys
sys.path.append('./')
sys.path.append('../')

from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
import torch

from torchvision.utils import save_image
import optim.adamw2
from safetensors.torch import load_file
from torchvision import transforms
from omegaconf import OmegaConf
import os
import numpy as np
import json
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument(
        "--sara_path",
        required=True,
        type=str,
        help="Path to the saved sara checkpoint.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Path to the saved sara checkpoint.",
    )
    parser.add_argument(
            "--pretrained_model_name_or_path",
            type=str,
            default='stabilityai/stable-diffusion-2',
            help="Path to pretrained model or model identifier from huggingface.co/models.",
        )
    parser.add_argument(
            "--output_dir",
            type=str,
            default='samples',
            help="Path to save the results",
        )

    def load_config(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config


    args = parser.parse_args()
    dataset_config = load_config(args.config)
    prefix_name=dataset_config['prefix_name']

    prompt_config = OmegaConf.load('../evaluation/large_scale_prompts.yaml')
    validation_prompts=prompt_config['prompt']

    pipeline = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path,revision=None, torch_dtype=torch.float32, safety_checker=None)

    ckpt=torch.load(args.sara_path)
    if 'threshold' in ckpt:
        threshold=ckpt['threshold']
        optimizer_cls = optim.adamw2.AdamW
        optimizer = optimizer_cls(pipeline.unet.parameters(),threshold=threshold)
    else:
        if args.threshold is None:
            raise "Please set a threshold for the loading ckpts."
        optimizer_cls = optim.adamw2.AdamW
        optimizer = optimizer_cls(pipeline.unet.parameters(), threshold=args.threshold)
        optimizer.load_params(ckpt)

    pipeline = pipeline.to('cuda')
    generator = torch.Generator(device=pipeline.unet.device).manual_seed(42)
    transforms=transforms.ToTensor()
    os.makedirs(args.output_dir, exist_ok=True)


    bs,cnt=8,0
    images,prompts = [],[]
    with torch.no_grad():
        for idx,validation_prompt in enumerate(validation_prompts):
            prompts.append('%s style, '%prefix_name+validation_prompt)
            if len(prompts)==bs or idx==len(validation_prompts)-1:
                image = pipeline(prompts, generator=generator).images
                for j in range(len(image)):
                    save_prompt=validation_prompts[cnt].replace(' ',"_")[:50]
                    save_image(transforms(image[j]), '%s/%d-%s.jpg'%(args.output_dir,cnt,save_prompt))
                    cnt+=1
                prompts=[]

