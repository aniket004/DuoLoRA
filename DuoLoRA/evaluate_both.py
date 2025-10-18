import argparse
import glob
import json
import os
import warnings
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from packaging import version
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm
import wandb

class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, append=False, prefix='A photo depicts'):
        self.data = data
        self.prefix = ''
        if append:
            self.prefix = prefix
            if self.prefix[-1] != ' ':
                self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)

def Convert(image):
    return image.convert("RGB")

class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            Convert,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        try:
            image = Image.open(c_data)
            image = self.preprocess(image)
        except UnidentifiedImageError:
            print(f"Error: Cannot identify image file '{c_data}'. Skipping this image.")
            image = torch.zeros(3, 224, 224)  # Return a dummy tensor for invalid images
        return {'image': image}

    def __len__(self):
        return len(self.data)

class DINOImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(256, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            Convert,
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        try:
            image = Image.open(c_data)
            image = self.preprocess(image)
        except UnidentifiedImageError:
            print(f"Error: Cannot identify image file '{c_data}'. Skipping this image.")
            image = torch.zeros(3, 224, 224)  # Return a dummy tensor for invalid images
        return {'image': image}

    def __len__(self):
        return len(self.data)

def extract_all_captions(captions, model, device, batch_size=256, num_workers=8, append=False):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions, append=append),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in tqdm(data):
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features

def extract_all_images(images, model, datasetclass, device, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        datasetclass(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in tqdm(data):
            b = b['image'].to(device)
            if hasattr(model, 'encode_image'):
                if device == 'cuda':
                    b = b.to(torch.float16)
                all_image_features.append(model.encode_image(b).cpu().numpy())
            else:
                all_image_features.append(model(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features

def get_clip_score(model, images, candidates, device, append=False, w=2.5):
    '''
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    '''
    if isinstance(images, list):
        # need to extract image features
        images = extract_all_images(images, model, device)

    candidates = extract_all_captions(candidates, model, device, append=append)

    # as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse('1.21'):
        images = sklearn.preprocessing.normalize(images, axis=1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        images = images / np.sqrt(np.sum(images ** 2, axis=1, keepdims=True))
        candidates = candidates / \
            np.sqrt(np.sum(candidates ** 2, axis=1, keepdims=True))

    per = w * np.clip(np.sum(images * candidates, axis=1), 0, None)
    return np.mean(per), per, candidates

def clipeval(image_dir, candidates_json, device):
    image_paths = [os.path.join(image_dir, path) for path in os.listdir(image_dir)
                   if path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG'))]
    image_ids = [Path(path).stem for path in image_paths]
    with open(candidates_json) as f:
        candidates = json.load(f)
    candidates = [candidates[cid] for cid in image_ids]

    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    image_feats = extract_all_images(
        image_paths, model, CLIPImageDataset, device, batch_size=64, num_workers=8)

    _, per_instance_image_text, _ = get_clip_score(
        model, image_feats, candidates, device)

    scores = {image_id: {'CLIPScore': float(clipscore)}
              for image_id, clipscore in
              zip(image_ids, per_instance_image_text)}
    print('CLIPScore: {:.4f}'.format(
        np.mean([s['CLIPScore'] for s in scores.values()])))

    return np.mean([s['CLIPScore'] for s in scores.values()]), np.std([s['CLIPScore'] for s in scores.values()])

def clipeval_image(image_dir, image_dir_ref, device):
    image_paths = [os.path.join(image_dir, path) for path in os.listdir(image_dir)
                   if path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG'))]
    image_paths_ref = [os.path.join(image_dir_ref, path) for path in os.listdir(image_dir_ref)
                       if path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG'))]

    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    image_feats = extract_all_images(
        image_paths, model, CLIPImageDataset, device, batch_size=64, num_workers=8)

    image_feats_ref = extract_all_images(
        image_paths_ref, model, CLIPImageDataset, device, batch_size=64, num_workers=8)

    image_feats = image_feats / \
        np.sqrt(np.sum(image_feats ** 2, axis=1, keepdims=True))
    image_feats_ref = image_feats_ref / \
        np.sqrt(np.sum(image_feats_ref ** 2, axis=1, keepdims=True))
    res = image_feats @ image_feats_ref.T
    return np.mean(res)

def dinoeval_image(image_dir, image_dir_ref, device):
    image_paths = [os.path.join(image_dir, path) for path in os.listdir(image_dir)
                   if path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG'))]
    image_paths_ref = [os.path.join(image_dir_ref, path) for path in os.listdir(image_dir_ref)
                       if path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG'))]

    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)
    model.eval()

    image_feats = extract_all_images(
        image_paths, model, DINOImageDataset, device, batch_size=64, num_workers=8)

    image_feats_ref = extract_all_images(
        image_paths_ref, model, DINOImageDataset, device, batch_size=64, num_workers=8)

    image_feats = image_feats / \
        np.sqrt(np.sum(image_feats ** 2, axis=1, keepdims=True))
    image_feats_ref = image_feats_ref / \
        np.sqrt(np.sum(image_feats_ref ** 2, axis=1, keepdims=True))
    res = image_feats @ image_feats_ref.T
    return np.mean(res)

def get_nested_folders(root_dir):
    nested_folders = []
    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            nested_folders.append(os.path.join(root, dir))
    return nested_folders

def process_content_similarity(content, args, device, max_images=10000):
    content_name = content.split('/')[-1]
    source_content_per_subject = os.path.join(args.source_content_image_path, content)
    target_content_per_subject = os.path.join(args.eval_dir, content_name)
    all_folders_per_subject = get_nested_folders(target_content_per_subject)

    all_dino_sim = []
    image_count = 0
    folder_name =[]

    for tar in all_folders_per_subject:
        image_paths = [os.path.join(tar, f) for f in os.listdir(tar) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG'))]
               
        if image_count >= max_images:
            break
        try:
            #breakpoint()
            sim = dinoeval_image(source_content_per_subject, tar, device)  # Pass directory instead of individual image file
            all_dino_sim.append(sim)
            folder_name.append(tar.split('/')[-2])
            image_count += 1
        except (ValueError, UnidentifiedImageError) as e:
            print(f"Error processing content {content_name}: {e}")
            continue

    print('DINO sim', all_dino_sim)
    print('style name', folder_name)
    return np.mean(all_dino_sim), content_name

def process_style_similarity(style, content_names_list, args, device, max_images=10000):
    style_name = style.split('/')[-1]
    source_style_per_subject = os.path.join(args.source_style_image_path, style_name)
    all_clip_style_sim = []
    image_count = 0

    for content_str in content_names_list:
        target_content_per_subject = os.path.join(args.eval_dir, content_str, style_name)
        all_folders_per_style = get_nested_folders(target_content_per_subject)

        for tar in all_folders_per_style:
            image_paths = [os.path.join(tar, f) for f in os.listdir(tar) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG'))]

            if image_count >= max_images:
                break
            try:
                sim = clipeval_image(source_style_per_subject, tar, device)  # Pass directory instead of individual image file
                all_clip_style_sim.append(sim)
                image_count += 1
            except (ValueError, UnidentifiedImageError) as e:
                print(f"Error processing style {style_name} for content {content_str}: {e}")
                continue


    return np.mean(all_clip_style_sim), style_name

def main():
    parser = argparse.ArgumentParser("metric", add_help=True)
    parser.add_argument("--source_content_image_path", type=str, default="PATH/ziplora_compare_set/objects")
    parser.add_argument("--source_style_image_path", type=str, default="PATH/ziplora_compare_set/styles")
    parser.add_argument("--eval_dir", type=str, default="PATH/output_content_style_ziploras_lambda_0.01_images")
    parser.add_argument("--max_images", type=int, default=10000, help="Maximum number of images to process")
    args = parser.parse_args()
    device = 'cuda'

    all_content = os.listdir(args.source_content_image_path)
    all_style = os.listdir(args.source_style_image_path)

    avg_per_object = []
    content_names_list = []
    for content in tqdm(all_content, desc="Processing Content Similarity"):
        avg_sim, content_name = process_content_similarity(content, args, device, args.max_images)
        avg_per_object.append(avg_sim)
        content_names_list.append(content_name)
        
        print('running dino sim per object:', avg_per_object)
        print('running content_names_list:', content_names_list)
        print('running dino eval sim:', np.nanmean(avg_per_object))


    content_names_list = all_content
    avg_per_style = []
    style_names_list = []
    for style in tqdm(all_style, desc="Processing Style Similarity"):
        avg_sim, style_name = process_style_similarity(style, content_names_list, args, device, args.max_images)
        avg_per_style.append(avg_sim)
        style_names_list.append(style_name)
        
        print('running clip style sim per style:', avg_per_style)
        print('running style_names_list:', style_names_list)
        print('running clip style eval sim:', np.nanmean(avg_per_style))


    print('############Final results######################## ')
    print('avg dino sim per object:', avg_per_object)
    print('content_names_list:', content_names_list)
    print('dino eval sim:', np.nanmean(avg_per_object)) 
    
    print('avg clip style sim per style:', avg_per_style)
    print('style_names_list:', style_names_list)
    print('clip style eval sim:', np.nanmean(avg_per_style))

    wandb.log({
        'max_images' : args.max_images,
        'dino eval sim': np.nanmean(avg_per_object),
        'clip style eval sim' : np.nanmean(avg_per_style)
    })

if __name__ == "__main__":
    main()
