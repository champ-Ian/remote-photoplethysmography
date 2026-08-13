# CNN Face Recognition Model

This folder contains a small, standalone CNN pipeline for face vs non-face classification.

## Structure
- `model.py` — CNN architecture
- `train.py` — trains with synthetic random images so it works even without a dataset
- `predict.py` — loads a saved model and predicts on one image

## Train the model

```bash
cd /Users/ianba/Desktop/Python/Heart Beat Detetection/remote-photoplethysmography
python cnn_facial_recognition_model/train.py --epochs 5 --batch_size 16
```

## Predict on a custom image

```bash
python cnn_facial_recognition_model/predict.py --image_path /path/to/image.jpg
```

## Predict on a random synthetic photo

```bash
python cnn_facial_recognition_model/predict.py
```

This model is intentionally simple and uses random initialization. It is meant as a base model for experimentation, not a production face-recognition system.
