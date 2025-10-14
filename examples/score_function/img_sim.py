# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import json
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import re, io
from typing import Dict
import torch, os
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

# device = "cuda"
# dino_processor = AutoImageProcessor.from_pretrained(
#     '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhengkaipeng-240108120123/dinov2-large'
# )
# dino_model = AutoModel.from_pretrained(
#     '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhengkaipeng-240108120123/dinov2-large'
# ).to(device).eval()
# clip_model = CLIPModel.from_pretrained(
#     "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhengkaipeng-240108120123/clip-vit-large-patch14-336"
# ).to(device).eval()
# clip_processor = CLIPProcessor.from_pretrained(
#     "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhengkaipeng-240108120123/clip-vit-large-patch14-336"
# )

def extract_code_from_text(text):
    if "```python" in text:
        code_blocks = re.findall(r'```python(.*?)```', text, re.DOTALL)
        combined_code = '\n'.join(block.strip() for block in code_blocks)
        return combined_code
    elif "```" in text:
        code_blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
        combined_code = '\n'.join(block.strip() for block in code_blocks)
        return combined_code
    else:
        return text

def remove_savefig(code):
    code = re.sub(r"plt\.savefig\(.*?\)", "", code)
    code = re.sub(r"plt\.close\(.*?\)", "", code)
    return code

def calculate_ssim(img1_path, img2_path):

    img1 = np.array(Image.open(img1_path).convert('L'))
    img2 = np.array(Image.open(img2_path).convert('L'))


    # Resize images to the same dimensions if necessary
    if img1.shape != img2.shape:
        height = min(img1.shape[0], img2.shape[0])
        width = min(img1.shape[1], img2.shape[1])
        img1 = np.array(Image.fromarray(img1).resize((width, height)))
        img2 = np.array(Image.fromarray(img2).resize((width, height)))

    # Compute SSIM; data_range is set based on the image pixel intensity range
    ssim_index = ssim(img1, img2, data_range=img1.max() - img1.min())
    if math.isnan(ssim_index):
        return 0.0
    else:
        return ssim_index
    
def calculate_psnr(image1_path, image2_path):
    img1 = np.array(Image.open(image1_path).convert('RGB'))
    img2 = np.array(Image.open(image2_path).convert('RGB'))

    if img1.shape != img2.shape:
        height = min(img1.shape[0], img2.shape[0])
        width = min(img1.shape[1], img2.shape[1])
        img1 = np.array(Image.fromarray(img1).resize((width, height)))
        img2 = np.array(Image.fromarray(img2).resize((width, height)))
    mse_val = np.mean((img1 - img2) ** 2)
    if mse_val == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_value = 20 * math.log10(max_pixel / math.sqrt(mse_val))
    return psnr_value

def hamming_similarity(image1_path, image2_path, hash_size=8):

    import imagehash
    hash1 = imagehash.average_hash(Image.open(image1_path), hash_size=hash_size)
    hash2 = imagehash.average_hash(Image.open(image2_path), hash_size=hash_size)
    distance = hash1 - hash2
    similarity = 1 - (distance / (hash_size * hash_size))
    return similarity

def get_dino_score(image1_path, image2_path):
    img1 = Image.open(image1_path).convert('RGB')
    img2 = Image.open(image2_path).convert('RGB')
    inputs1 = dino_processor(images=img1, return_tensors='pt').to(device)
    inputs2 = dino_processor(images=img2, return_tensors='pt').to(device)
    with torch.no_grad():
        vision_outputs1 = dino_model(**inputs1)
        vision_outputs2 = dino_model(**inputs2)
    embed1 = vision_outputs1.last_hidden_state.mean(dim=1)
    embed2 = vision_outputs2.last_hidden_state.mean(dim=1)
    cosine_sim = torch.nn.CosineSimilarity(dim=0)(embed1[0], embed2[0]).item()
    # 转换到百分制
    similarity = 100 * cosine_sim
    return round(similarity, 4)


def get_clip_score(image1_path, image2_path):
    img1 = Image.open(image1_path).convert('RGB')
    img2 = Image.open(image2_path).convert('RGB')
    inputs1 = clip_processor(images=img1, return_tensors='pt', padding=True).to(device)
    inputs2 = clip_processor(images=img2, return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        vision_outputs1 = clip_model.vision_model(**inputs1)
        vision_outputs2 = clip_model.vision_model(**inputs2)
    image_embeds1 = vision_outputs1[1]
    image_embeds2 = vision_outputs2[1]
    image_embeds1 = clip_model.visual_projection(image_embeds1)
    image_embeds2 = clip_model.visual_projection(image_embeds2)
    image_embeds1 = image_embeds1 / image_embeds1.norm(dim=-1, keepdim=True)
    image_embeds2 = image_embeds2 / image_embeds2.norm(dim=-1, keepdim=True)
    similarity = (100.0 * (image_embeds1 @ image_embeds2.T)).sum(dim=-1)
    return round(similarity.item(), 4)
from mathruler.grader import extract_boxed_content, grade_answer




def accuracy_reward(predict_str: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict_str)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


# def compute_score(predict_str: str, ground_truth: str, format_weight: float = 0.1) -> Dict[str, float]:
#     code = extract_code_from_text(predict_str)
#     modified_code = remove_savefig(code)
#     new_name = os.path.splitext(os.path.basename(ground_truth))[0]
#     generated_image_path = f'/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhengkaipeng-240108120123/weilai/codes/codemllm/temp/{new_name}.jpg'
#     # print(modified_code)
#     # print('======')
#     try:
#         local_namespace = {}
#         compiled_code = compile(modified_code.replace('plt.close()', '').replace('plt.close(fig)', ''), '<string>', 'exec')
#         exec(compiled_code, local_namespace)
#         plt.savefig(generated_image_path)
#         plt.cla()
#         plt.close("all")
#         sim_score = calculate_ssim(ground_truth, generated_image_path)
#         psnr_score = calculate_psnr(ground_truth, generated_image_path)
#         hash_score = hamming_similarity(ground_truth, generated_image_path)
#         dino_score = get_dino_score(ground_truth, generated_image_path)
#         clip_score = get_clip_score(ground_truth, generated_image_path)
#         avg_score = clip_score + dino_score + 10 * sim_score + 10 * hash_score + 6 * psnr_score

#     except Exception as e:
#         print(e)
#         clip_score = 40
#         dino_score = 40
#         hash_score = 0
#         psnr_score = 0
#         sim_score = 0
#         avg_score = clip_score + dino_score + 10 * sim_score + 10 * hash_score + 6 * psnr_score

#     return {
#         "overall": avg_score,
#         "clip": clip_score,
#         "dino": dino_score,
#     }

# def compute_score(predict_str: str, ground_truth: str, format_weight: float = 0.1) -> Dict[str, float]:
#     code = extract_code_from_text(predict_str)
#     modified_code = remove_savefig(code)
#     new_name = os.path.splitext(os.path.basename(ground_truth))[0]
#     generated_image_path = f'/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhengkaipeng-240108120123/weilai/codes/codemllm/temp/{new_name}.jpg'
#     # print(modified_code)
#     # print('======')
#     try:
#         local_namespace = {}
#         compiled_code = compile(modified_code.replace('plt.close()', '').replace('plt.close(fig)', ''), '<string>', 'exec')
#         exec(compiled_code, local_namespace)
#         plt.savefig(generated_image_path)
#         plt.cla()
#         plt.close("all")
#         sim_score = calculate_ssim(ground_truth, generated_image_path)
#         psnr_score = calculate_psnr(ground_truth, generated_image_path)
#         hash_score = hamming_similarity(ground_truth, generated_image_path)
   
#         avg_score = 10 * sim_score + 10 * hash_score + 6 * psnr_score

#     except Exception as e:
#         print(e)
#         hash_score = 0
#         psnr_score = 0
#         sim_score = 0
#         avg_score = 10 * sim_score + 10 * hash_score + 6 * psnr_score

#     return {
#         "overall": avg_score,
#     }

# def compute_score(predict_str: str, ground_truth: str, format_weight: float = 0.1) -> Dict[str, float]:
#     code = extract_code_from_text(predict_str)
#     modified_code = remove_savefig(code)

#     try:
#         # Run the code in a local namespace
#         local_namespace = {}
#         compiled_code = compile(modified_code.replace('plt.close()', '').replace('plt.close(fig)', ''), '<string>', 'exec')
#         exec(compiled_code, local_namespace)

#         # Save the generated plot into memory
#         buf = io.BytesIO()
#         plt.savefig(buf, format='jpeg')
#         buf.seek(0)
#         gen_img = Image.open(buf).convert("RGB")

#         # Load ground truth image from file
#         gt_img = Image.open(ground_truth).convert("RGB")

#         # Close and clean up
#         plt.cla()
#         plt.close("all")

#         # Compute similarity scores
#         sim_score = calculate_ssim(gt_img, gen_img)
#         psnr_score = calculate_psnr(gt_img, gen_img)
#         hash_score = hamming_similarity(gt_img, gen_img)

#         avg_score = 10 * sim_score + 10 * hash_score + 6 * psnr_score

#     except Exception as e:
#         print(e)
#         hash_score = 0
#         psnr_score = 0
#         sim_score = 0
#         avg_score = 0

#     return {
#         "overall": avg_score,
#     }
from codebleu import calc_codebleu

def compute_score(predict_str: str, ground_truth: str, format_weight: float = 0.1) -> Dict[str, float]:
    # try:
    #     # Run the code in a local namespace
    #     code = extract_code_from_text(predict_str)
    #     modified_code = remove_savefig(code)
    #     local_namespace = {}
    #     compiled_code = compile(modified_code.replace('plt.close()', '').replace('plt.close(fig)', '').replace('exit()',''), '<string>', 'exec')
    #     exec(compiled_code, local_namespace)
    #     can_run = 1
    # except Exception as e:
    #     print(e)
    #     can_run = 0
    code = extract_code_from_text(predict_str)
    # modified_code = remove_savefig(code)
    sim_score = calc_codebleu([ground_truth], [code], lang='python', weights=(0.1, 0.1, 0.4, 0.4), tokenizer=None)['codebleu']
    # total_score = 0.7 * sim_score + 0.3 * can_run
    return {
        "overall": float(sim_score),
        "sim_score": float(sim_score),
        # "can_run": can_run
    }
