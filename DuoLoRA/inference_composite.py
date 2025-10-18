import argparse
#import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline
from ziplora_pytorch.utils import insert_ziplora_to_unet
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="pretrained model path",
    )
    parser.add_argument(
        "--ziplora_name_or_path", type=str, required=True, help="ziplora path"
    )
    parser.add_argument("--prompt", type=str,default="a sbu dog and szn cat")
    parser.add_argument("--out_dir", type=str, required=True, help="output dir")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = StableDiffusionXLPipeline.from_pretrained(args.pretrained_model_name_or_path)
    pipeline.unet = insert_ziplora_to_unet(pipeline.unet, args.ziplora_name_or_path)
    pipeline.to(device=device, dtype=torch.float16)
    for i in range(5):
        image = pipeline(prompt=args.prompt).images[0]
        image.save(f'{args.out_dir}/test_w_rev_cycle_{i}.png')


if __name__ == "__main__":
    main()