import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import FaceCNN


class SyntheticFaceDataset(Dataset):
    def __init__(self, n_samples=200, image_size=64, seed=42):
        self.rng = np.random.default_rng(seed)
        self.samples = []
        self.labels = []

        for _ in range(n_samples):
            is_face = self.rng.random() > 0.5
            img = self.make_random_image(image_size, is_face=is_face)
            self.samples.append(img)
            self.labels.append(1 if is_face else 0)

    def make_random_image(self, size, is_face=True):
        img = self.rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        if not is_face:
            return img

        center_x = int(self.rng.integers(size * 0.35, size * 0.65))
        center_y = int(self.rng.integers(size * 0.40, size * 0.60))
        axes_x = int(self.rng.integers(size * 0.15, size * 0.25))
        axes_y = int(self.rng.integers(size * 0.18, size * 0.32))

        face = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.ellipse(face, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, (200, 170, 150), -1)

        eye1 = (center_x - axes_x // 3, center_y - axes_y // 4)
        eye2 = (center_x + axes_x // 3, center_y - axes_y // 4)
        cv2.circle(face, eye1, max(3, axes_x // 10), (255, 255, 255), -1)
        cv2.circle(face, eye2, max(3, axes_x // 10), (255, 255, 255), -1)
        cv2.circle(face, eye1, max(1, axes_x // 18), (0, 0, 0), -1)
        cv2.circle(face, eye2, max(1, axes_x // 18), (0, 0, 0), -1)

        nose_y = center_y + axes_y // 8
        cv2.line(face, (center_x, center_y + axes_y // 6), (center_x, nose_y + axes_y // 4), (120, 90, 80), 2)
        cv2.ellipse(face, (center_x, center_y + axes_y // 2), (axes_x // 3, axes_y // 8), 0, 0, 180, (160, 100, 100), 2)

        img = cv2.GaussianBlur(img, (3, 3), 0)
        mask = np.mean(face, axis=2) > 10
        img[mask] = face[mask]
        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = self.samples[idx].astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        label = self.labels[idx]
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output', type=str, default='cnn_facial_recognition_model/face_model.pth')
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    dataset = SyntheticFaceDataset(n_samples=300, image_size=64)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FaceCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f'Epoch {epoch}/{args.epochs} | loss={loss:.4f} | acc={acc:.4f}')

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f'Model saved to {output_path}')


if __name__ == '__main__':
    main()
