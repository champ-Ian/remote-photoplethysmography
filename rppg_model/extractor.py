import numpy as np
from scipy.signal import butter, filtfilt

try:
    from .utils import read_video_frames, roi_mean_color
except ImportError:  # pragma: no cover - direct script execution fallback
    from utils import read_video_frames, roi_mean_color


def bandpass_filter(x, fs, low=0.7, high=4.0, order=4):
    ny = 0.5 * fs
    lown = low / ny
    highn = high / ny
    b, a = butter(order, [lown, highn], btype='band')
    y = filtfilt(b, a, x)
    return y


def color_trace_from_video(path, resize=(160, 120), max_frames=None):
    frames = read_video_frames(path, max_frames=max_frames, resize=resize)
    if len(frames) == 0:
        return np.zeros((0, 3)), 0
    colors = []
    for f in frames:
        colors.append(roi_mean_color(f))
    colors = np.array(colors)  # N x 3 (R,G,B)
    # estimate frame rate via VideoCapture in read_video_frames isn't returned; default to 30
    # user can modify this if different fps
    fs = 30.0
    return colors, fs


def get_bvp_signal_from_trace(rgb_ts, fs=30.0):
    # Use the green channel as a simple rPPG proxy, then bandpass
    if rgb_ts.shape[0] == 0:
        return np.array([])
    green = rgb_ts[:, 1]
    # detrend
    green = (green - np.mean(green)) / (np.std(green) + 1e-8)
    bvp = bandpass_filter(green, fs)
    return bvp


def estimate_hr_via_fft(bvp, fs=30.0, min_bpm=40, max_bpm=240):
    if len(bvp) == 0:
        return None
    n = len(bvp)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft = np.abs(np.fft.rfft(bvp))
    idx = np.where((freqs >= min_bpm / 60.0) & (freqs <= max_bpm / 60.0))[0]
    if len(idx) == 0:
        return None
    peak = idx[np.argmax(fft[idx])]
    freq = freqs[peak]
    bpm = freq * 60.0
    return float(bpm)
