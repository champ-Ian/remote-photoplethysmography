import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from model import FaceCNN


def make_random_face_like_image(size=64):
    rng = np.random.default_rng()
    img = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    cx = size // 2
    cy = size // 2
    rx = size // 3
    ry = size // 2
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (200, 170, 150), -1)
    cv2.circle(img, (cx - rx // 3, cy - ry // 5), 5, (255, 255, 255), -1)
    cv2.circle(img, (cx + rx // 3, cy - ry // 5), 5, (255, 255, 255), -1)
    cv2.circle(img, (cx - rx // 3, cy - ry // 5), 2, (0, 0, 0), -1)
    cv2.circle(img, (cx + rx // 3, cy - ry // 5), 2, (0, 0, 0), -1)
    cv2.line(img, (cx, cy), (cx, cy + ry // 3), (120, 80, 80), 2)
    cv2.ellipse(img, (cx, cy + ry // 2), (rx // 3, ry // 8), 0, 0, 180, (150, 100, 100), 2)
    return img


def load_image(path):
    if path is None:
        return make_random_face_like_image(), 'random synthetic face image'

    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f'Could not read image: {path}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, path


def preprocess(image):
    image = cv2.resize(image, (64, 64))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0)


def classify_patch(model, image_patch, device):
    tensor = preprocess(image_patch).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = logits.argmax(dim=1).item()
        pred_label = 'face' if pred_idx == 1 else 'not_face'
        conf = float(probs[pred_idx].item())
    return pred_label, conf, probs.cpu().numpy()


def build_face_mask(model, image, device='cpu', patch_size=64, stride=16):
    image_h, image_w = image.shape[:2]
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    for y in range(0, max(1, image_h - patch_size + 1), stride):
        for x in range(0, max(1, image_w - patch_size + 1), stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = cv2.resize(patch, (patch_size, patch_size))
            label, conf, probs = classify_patch(model, patch, device)
            if label == 'face' and conf > 0.55:
                mask[y:y + patch_size, x:x + patch_size] = 255
    if np.count_nonzero(mask) > 0:
        mask = cv2.medianBlur(mask, 5)
    return mask


def predict(model_path='cnn_facial_recognition_model/face_model.pth', image_path=None, device='cpu', output_mask_path=None):
    device = torch.device(device)
    model = FaceCNN().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    image, source = load_image(image_path)
    tensor = preprocess(image).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = logits.argmax(dim=1).item()
        pred_label = 'face' if pred_idx == 1 else 'not_face'
        conf = float(probs[pred_idx].item())

    print(f'Input: {source}')
    print(f'Prediction: {pred_label} (confidence={conf:.3f})')
    print(f'Class probabilities: face={probs[1].item():.3f}, not_face={probs[0].item():.3f}')

    mask = build_face_mask(model, image, device=device)
    if output_mask_path is None and source != 'random synthetic face image':
        output_mask_path = str(Path(source).with_name(Path(source).stem + '_mask.png'))
    elif output_mask_path is None:
        output_mask_path = 'random_face_mask.png'

    overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    color_mask = np.zeros_like(overlay)
    color_mask[mask > 0] = (0, 255, 0)
    overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.5, 0)
    cv2.imwrite(output_mask_path, overlay)
    print(f'Mask overlay saved to {output_mask_path}')

    return pred_label, conf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='cnn_facial_recognition_model/face_model.pth')
    parser.add_argument('--image_path', default=None, help='Optional path to a real image. If omitted, a random synthetic face-like image is used.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output_mask_path', default=None, help='Optional custom path for the image with a face mask overlay.')
    args = parser.parse_args()

    predict(args.model_path, args.image_path, args.device, args.output_mask_path)
