import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import os


def load_model(checkpoint_path, device='cpu'):
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def preprocess_image(image_path, image_size=(256, 512)):
    image = Image.open(image_path).convert('RGB').resize(image_size)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(np.transpose(img_array, (2, 0, 1))).unsqueeze(0)
    return image, img_tensor


def predict_mask(model, img_tensor, threshold=0.5):
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output)
        mask = (prob > threshold).float()
    return mask.squeeze().numpy()


def visualize_prediction(original_image, mask, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')

    axes[2].imshow(original_image)
    axes[2].imshow(mask, cmap='Reds', alpha=0.4)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model('models/best_model_v2_safe.pth', device=device)

    input_dir = 'data/sample_test_images'
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        original_image, img_tensor = preprocess_image(img_path)

        mask = predict_mask(model, img_tensor)

        save_path = os.path.join(output_dir, f"result_{img_file.replace('.jpg', '.png')}")
        visualize_prediction(original_image, mask, save_path)

    print(f"\nDone! Check the '{output_dir}' folder for results.")


import sys
sys.path.append(os.path.dirname(__file__))
from postprocess import draw_lane_curves, compute_lane_offset

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model('models/best_model_v2_safe.pth', device=device)

    input_dir = 'data/sample_test_images'
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        original_image, img_tensor = preprocess_image(img_path)

        mask = predict_mask(model, img_tensor)

        # Post-process: fit curves and compute offset
        curved_image, fitted_curves = draw_lane_curves(original_image, mask)
        offset = compute_lane_offset(fitted_curves, mask.shape[1], mask.shape[0])

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(original_image)
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(curved_image)
        offset_text = f"Offset: {offset:.1f}px" if offset is not None else "Offset: N/A"
        axes[1].set_title(f'Fitted Lanes ({offset_text})')
        axes[1].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"curves_{img_file.replace('.jpg', '.png')}")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path} | {offset_text}")

    print(f"\nDone! Check the '{output_dir}' folder for results.")



