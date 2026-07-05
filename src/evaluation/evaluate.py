import os
import torch
import numpy as np
import cv2
from PIL import Image


def compute_iou_dice(pred_mask, gt_mask, threshold=0.5):
    pred_bin = (pred_mask > threshold).astype(np.float32)
    gt_bin = (gt_mask > 0).astype(np.float32)

    intersection = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)

    dice = (2 * intersection + 1e-6) / (pred_bin.sum() + gt_bin.sum() + 1e-6)

    return iou, dice


def evaluate_on_test_set(model, entries, test_clips_dir, device, image_size=(256, 512)):
    model.eval()
    total_iou, total_dice, count = 0.0, 0.0, 0

    for entry in entries:
        img_path = os.path.join(test_clips_dir, entry['raw_file'])
        gt_mask_path = entry['mask_path']

        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert('RGB').resize(image_size)
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(np.transpose(img_array, (2, 0, 1))).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()

        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask_resized = cv2.resize(gt_mask, image_size)

        iou, dice = compute_iou_dice(pred_prob, gt_mask_resized)
        total_iou += iou
        total_dice += dice
        count += 1

        if count % 200 == 0:
            print(f"Processed {count} test images...")

    print(f"\n=== Final Test Set Results ===")
    print(f"Total images evaluated: {count}")
    print(f"Mean IoU: {total_iou/count:.4f}")
    print(f"Mean Dice: {total_dice/count:.4f}")

    return total_iou / count, total_dice / count


if __name__ == "__main__":
   
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.evaluation.generate_gt_masks import generate_all_masks

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_json_path = '/content/data/TUSimple/test_label.json'
    gt_mask_dir = '/content/data/TUSimple/test_set/gt_masks'
    test_clips_dir = '/content/data/TUSimple/test_set'

    entries = generate_all_masks(test_json_path, gt_mask_dir)

   
