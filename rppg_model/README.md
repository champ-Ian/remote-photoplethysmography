rPPG Model — Predict Heart Rate from Video

Overview
- Small pipeline to extract color traces from video, generate pseudo-labels using a signal-processing rPPG estimator, and train a simple 1D-CNN to predict heart rate (BPM) from short video segments.

Usage
1. Install dependencies: `pip install -r rppg_model/requirements.txt`
2. Prepare a folder of video files (mp4/avi). Example: `videos/subject1.mp4`
3. Run training (example):
```
python rppg_model/train.py --input_dir videos --output_model rppg_model/checkpoint.pth --epochs 10
```

Notes
- This implementation uses pseudo-labels produced by a classical rPPG estimator (FFT on filtered green/CHROM signal). It does not use any ECG files for training.
- Designed to be a starting point; tune preprocessing, ROI extraction, and model architecture for improved results.
