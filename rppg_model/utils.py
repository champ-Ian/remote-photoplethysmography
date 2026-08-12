import cv2
import numpy as np


def read_video_frames(path, max_frames=None, resize=None):
    cap = cv2.VideoCapture(path)
    frames = []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if resize is not None:
            frame = cv2.resize(frame, resize)
        frames.append(frame)
        i += 1
        if max_frames and i >= max_frames:
            break
    cap.release()
    return np.array(frames)


def get_skin_mask(frame):
    # Use YCrCb color space for simple skin detection
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    (y, cr, cb) = cv2.split(img)
    mask = cv2.inRange(img, np.array((0, 133, 77)), np.array((255, 173, 127)))
    return mask


def roi_mean_color(frame, mask=None, fallback_size=0.2):
    if mask is None:
        mask = get_skin_mask(frame)
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    # compute mean color where mask > 0
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        # fallback to central crop
        h, w = frame.shape[:2]
        ch, cw = int(h * fallback_size), int(w * fallback_size)
        cy, cx = h // 2, w // 2
        crop = frame[cy - ch // 2: cy + ch // 2, cx - cw // 2: cx + cw // 2]
        mean = crop.reshape(-1, 3).mean(axis=0)
        return mean[::-1]  # return RGB
    vals = masked[ys, xs]
    mean = vals.mean(axis=0)
    # convert BGR to RGB
    return mean[::-1]
