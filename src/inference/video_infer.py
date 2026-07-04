import cv2
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import sys
import os

sys.path.append(os.path.dirname(__file__))
from predict import load_model, predict_mask
from postprocess import draw_lane_curves, compute_lane_offset


def process_video(video_path, output_path, model, device, image_size=(256, 512)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))

    print(f"Processing {frame_count} frames at {fps:.1f} FPS...")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR (OpenCV) -> RGB (PIL/model expects this)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb).resize(image_size)

        img_array = np.array(pil_image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(np.transpose(img_array, (2, 0, 1))).unsqueeze(0).to(device)

        mask = predict_mask(model, img_tensor)

        # Draw curves on the resized image
        curved_image, fitted_curves = draw_lane_curves(pil_image, mask)
        offset = compute_lane_offset(fitted_curves, mask.shape[1], mask.shape[0])

        # Resize annotated frame back to original video resolution
        curved_image_resized = cv2.resize(np.array(curved_image), (orig_width, orig_height))
        curved_bgr = cv2.cvtColor(curved_image_resized, cv2.COLOR_RGB2BGR)

        # Add offset text overlay
        if offset is not None:
            text = f"Offset: {offset:.1f}px"
            cv2.putText(curved_bgr, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3)

        out.write(curved_bgr)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Processed {frame_idx}/{frame_count} frames...")

    cap.release()
    out.release()
    print(f"Done! Saved to {output_path}")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model('models/best_model_v2_safe.pth', device=device)

    input_video = 'data/sample_video_short.mp4'   # change this to your video path
    output_video = 'outputs/annotated_video.mp4'
    os.makedirs('outputs', exist_ok=True)

    process_video(input_video, output_video, model, device)