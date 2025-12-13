import os
import sys
sys.path.append("sam2/")
from PIL import Image
import torch
import cv2
import numpy as np

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

input_folder = "cropped_hand_images/"    # folder containing your images
output_folder = "output/"   # folder to save segmented hands

os.makedirs(output_folder, exist_ok=True)

# Load SAM model (choose appropriate model type)
checkpoint = "<path to checkpoint>"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

def segment_hand(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    center_point = np.array([w//2, h//2])

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image_rgb)
        masks, scores, logits = predictor.predict(
            point_coords=center_point[None, :],
            point_labels=np.array([1]),
            multimask_output=False
        )
    
    mask = masks[0]  # take first mask
    segmented = image_rgb * mask[:, :, None]
    segmented = cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)
    
    return segmented

# Process all images
subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]
for subfolder in subfolders:
    subfolder_name = os.path.basename(subfolder)
    if int(subfolder_name) <= 3:
        continue
    output_subfolder = os.path.join(output_folder, subfolder_name)
    os.makedirs(output_subfolder, exist_ok=True)
    
    for fname in os.listdir(subfolder):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(subfolder, fname)
            output_path = os.path.join(output_subfolder, fname)
            segmented_image = segment_hand(input_path)
            cv2.imwrite(output_path, segmented_image)
            print(f"Processed and saved: {output_path}")