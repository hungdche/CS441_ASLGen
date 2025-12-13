import cv2
import mediapipe as mp
import os
import time
import argparse

def find_hand_bounding_box(image, results):
    '''
    compute a square bounding box around the detected hand landmarks with padding.
    '''
    
    if not results.multi_hand_landmarks:
        return None
    
    h, w, c = image.shape

    for hand_landmarks in results.multi_hand_landmarks:

        # unnormalized detected coords
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        # get an init bounding box
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # apply padding
        x_min, y_min = x_min - PADDING, y_min - PADDING
        x_max, y_max = x_max + PADDING, y_max + PADDING

        # ensure box is within image bounds
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)

        # compute center of the box
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2

        # get square bbox
        width = x_max - x_min
        height = y_max - y_min
        side = max(width, height) 

        sq_x_min = cx - side // 2
        sq_x_max = sq_x_min + side
        sq_y_min = cy - side // 2
        sq_y_max = sq_y_min + side

        # handle when square goes outside image bounds
        if sq_x_min < 0:
            sq_x_max += -sq_x_min
            sq_x_min = 0
        if sq_x_max > w:
            shift = sq_x_max - w
            sq_x_min -= shift
            sq_x_max = w
        if sq_y_min < 0:
            sq_y_max += -sq_y_min
            sq_y_min = 0
        if sq_y_max > h:
            shift = sq_y_max - h
            sq_y_min -= shift
            sq_y_max = h

        return (int(sq_x_min), int(sq_y_min), int(sq_x_max), int(sq_y_max))
    return None

if "__main__" == __name__:

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_resolution", type=int, default=512)
    parser.add_argument("--alphabet", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=50)
    args = parser.parse_args()

    TARGET_RESOLUTION = (args.target_resolution, args.target_resolution)
    SAVE_DIR = f"cropped_hand_images/{args.alphabet}"
    os.makedirs(SAVE_DIR, exist_ok=True)
    PADDING = 40

    # model to detect hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # webcam 
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    saved_img_count = 0
    while cap.isOpened():
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
                resized_image = cv2.resize(cropped_image, TARGET_RESOLUTION, interpolation=cv2.INTER_AREA)
                cv2.imshow("Resized Hand Pose", resized_image)

        # Handle key presses
        key = cv2.waitKey(5) & 0xFF
        
        # Save the resized image
        if key == ord('s'):
            if resized_image is not None:
                # random name
                timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1e9) % 1_000_000_000:09d}"
                filename = os.path.join(SAVE_DIR, f"hand_pose_{timestamp}.png")

                cv2.imwrite(filename, resized_image)
                print(f"Saved image: {filename}")

                saved_img_count += 1
                if saved_img_count >= args.num_images:
                    print(f"Reached target of {args.num_images} images. Exiting.")
                    break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()