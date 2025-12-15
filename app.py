import cv2
import mediapipe as mp
import os
import sys
sys.path.append("sam2/")

import time
import argparse
import imageio

import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from image_capture import find_hand_bounding_box
from models import SimpleCNN

# Load SAM model (choose appropriate model type)
checkpoint = "/home/hche/school/CS441/final_project/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

def remove_bg(image):
    h, w, _ = image.shape
    center_point = np.array([w//2, h//2])

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=center_point[None, :],
            point_labels=np.array([1]),
            multimask_output=False
        )
    
    mask = masks[0]  # take first mask
    segmented = image * mask[:, :, None]
    segmented = cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)
    return segmented

def load_gif(path, resize=None):
    gif = imageio.mimread(path)
    frames = []
    for f in gif:
        frame = np.array(f)
        if frame.shape[-1] == 4:  # RGBA → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        if resize is not None:
            frame = cv2.resize(frame, resize)
        
        # convert to BGR for cv2 display
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(frame)
    return frames


if "__main__" == __name__:

    parser = argparse.ArgumentParser()
    parser.add_argument("--alphabet", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--gen_model", type=str, default="vae", choices=["vae", "gan"])
    args = parser.parse_args()

    # model to detect hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # model to classify hand poses
    simple_cnn = SimpleCNN(num_classes=6).to("cuda")
    simple_cnn.load_state_dict(torch.load(args.ckpt))
    simple_cnn.eval()

    # display gif
    if args.gen_model == "gan":
        gif_path = f"generated_gif/transition_{args.alphabet}.gif"
    elif args.gen_model == "vae":
        gif_path = f"generated_gif/transition_{args.alphabet}_vae.gif"

    gif_frames = load_gif(gif_path)
    gif_idx = 0
    gif_len = len(gif_frames)

    cv2.namedWindow("Reference GIF", cv2.WINDOW_AUTOSIZE)

    # webcam 
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    saved_img_count = 0
    while cap.isOpened():
        gif_frame = gif_frames[gif_idx]
        cv2.imshow("Reference GIF", gif_frame)
        gif_idx = (gif_idx + 1) % gif_len

        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # flip frame 
        frame = cv2.flip(frame, 1)

        # grab image + try to detect hand
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        resized_image = None
        if results.multi_hand_landmarks:
            bbox = find_hand_bounding_box(frame, results)

            # if detect hand
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                cropped_image = frame[y_min:y_max, x_min:x_max]
                resized_image = cv2.resize(cropped_image, (128, 128), interpolation=cv2.INTER_AREA)

                # cv2.imshow("Resized Hand Pose", resized_image)

                # run inference to see if pose is correct 
                resized_image = remove_bg(resized_image)
                input_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).unsqueeze(0).float().to("cuda") / 255.0
                with torch.no_grad():
                    outputs = simple_cnn(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                    class_idx = predicted.item()
                    class_names = ['0', '1', '2', '3', '4', '5']  # adjust based on your classes
                    predicted_class = class_names[class_idx]
                    if predicted_class == args.alphabet:
                        cv2.putText(frame, f'Correct hand pose for {args.alphabet}', (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, f'Incorrect hand pose for {args.alphabet}', (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        
                    cv2.imshow("Webcam Feed", frame)
                    print(f'Predicted class: {predicted_class}')

        # Handle key presses
        key = cv2.waitKey(5) & 0xFF

    cap.release()
    cv2.destroyAllWindows()
    hands.close()