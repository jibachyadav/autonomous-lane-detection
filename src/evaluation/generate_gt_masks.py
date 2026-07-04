import json
import cv2
import numpy as np
import os


def generate_mask_from_lanes(lanes, h_samples, img_shape=(720, 1280)):
    mask = np.zeros(img_shape, dtype=np.uint8)
    for lane in lanes:
        points = [(x, y) for x, y in zip(lane, h_samples) if x != -2]
        for i in range(len(points) - 1):
            cv2.line(mask, points[i], points[i + 1], color=255, thickness=10)
    return mask


def generate_all_masks(test_json_path, gt_mask_dir):
    os.makedirs(gt_mask_dir, exist_ok=True)

    with open(test_json_path, 'r') as f:
        lines = f.readlines()

    print(f"Total test entries: {len(lines)}")

    entries = []
    for line in lines:
        entry = json.loads(line)
        lanes = entry['lanes']
        h_samples = entry['h_samples']
        raw_file = entry['raw_file']

        mask = generate_mask_from_lanes(lanes, h_samples)

        mask_filename = raw_file.replace('/', '_').replace('.jpg', '.png')
        mask_path = os.path.join(gt_mask_dir, mask_filename)
        cv2.imwrite(mask_path, mask)

        entries.append({'raw_file': raw_file, 'mask_path': mask_path})

    print(f"Generated {len(entries)} ground-truth masks")
    return entries


if __name__ == "__main__":
    # Note: run this on Colab/Kaggle where the TuSimple test set is available
    test_json_path = '/content/data/TUSimple/test_label.json'
    gt_mask_dir = '/content/data/TUSimple/test_set/gt_masks'
    generate_all_masks(test_json_path, gt_mask_dir)