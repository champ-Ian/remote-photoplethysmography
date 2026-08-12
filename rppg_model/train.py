import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

from extractor import color_trace_from_video, get_bvp_signal_from_trace, estimate_hr_via_fft
from model import RPPGNet


class TraceDataset(Dataset):
    def __init__(self, samples, window_size=256):
        self.data = []
        self.labels = []
        for rgb, fs, bpm in samples:
            n = rgb.shape[0]
            if n < window_size:
                continue
            for i in range(0, n - window_size + 1, window_size // 2):
                seg = rgb[i:i + window_size]
                # normalize per-segment
                seg = (seg - seg.mean(axis=0)) / (seg.std(axis=0) + 1e-8)
                self.data.append(torch.from_numpy(seg.astype(np.float32)))
                self.labels.append(torch.tensor(bpm, dtype=torch.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def build_pseudo_labeled_samples(input_dir, max_videos=None):
    samples = []
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if max_videos:
        files = files[:max_videos]
    for p in tqdm(files, desc='processing videos'):
        rgb, fs = color_trace_from_video(p)
        bvp = get_bvp_signal_from_trace(rgb, fs=fs)
        bpm = estimate_hr_via_fft(bvp, fs=fs)
        if bpm is None:
            continue
        samples.append((rgb, fs, bpm))
    return samples


def train_loop(dataloader, model, opt, loss_fn, device):
    model.train()
    running = 0.0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        # x is B x T x C
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        running += loss.item() * x.size(0)
    return running / len(dataloader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_model', default='rppg_model/checkpoint.pth')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_videos', type=int, default=None)
    args = parser.parse_args()

    samples = build_pseudo_labeled_samples(args.input_dir, max_videos=args.max_videos)
    if len(samples) == 0:
        print('No pseudo-labeled samples found. Add video files to the input directory.')
        return
    ds = TraceDataset(samples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RPPGNet(in_channels=3).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for ep in range(args.epochs):
        loss = train_loop(loader, model, opt, loss_fn, device)
        print(f'Epoch {ep+1}/{args.epochs} loss={loss:.4f}')

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    torch.save(model.state_dict(), args.output_model)
    print('Model saved to', args.output_model)


if __name__ == '__main__':
    main()
