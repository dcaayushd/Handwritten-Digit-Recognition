# Handwritten Digit Recognition System

A handwritten digit recognition system built with PyTorch, FastAPI, Docker, and pytest.

## Stack

- Python 3.10+
- PyTorch
- torchvision
- FastAPI
- Uvicorn
- Docker
- pytest

## Project Layout

```text
.
├── app/
│   ├── main.py
│   ├── config.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── data/
├── models/
├── tests/
├── .env.example
├── Dockerfile
├── README.md
└── requirements.txt
```

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Environment

Example `.env` values:

```env
MODEL_VERSION=v3
MODEL_PATH=models/mnist_cnn_v3.pt
LOG_LEVEL=INFO
MIN_PREDICTION_CONFIDENCE=0.55
MIN_PREDICTION_MARGIN=0.18
```

## Train

Train the current production model:

```bash
python -m app.train --version v3 --architecture mnist_resnet --epochs 14 --batch-size 128 --learning-rate 0.0015
```

This downloads MNIST into `data/`, trains the model, evaluates it, and saves the checkpoint to `models/`.

## Fine-Tune On Personal Handwriting

Create labeled folders:

```text
data/user_digits/
├── 0/
├── 1/
├── 2/
├── 3/
├── 4/
├── 5/
├── 6/
├── 7/
├── 8/
└── 9/
```

Then fine-tune from the current checkpoint:

```bash
python -m app.train \
  --version v4_personal \
  --architecture mnist_resnet \
  --epochs 6 \
  --batch-size 64 \
  --learning-rate 0.0005 \
  --fine-tune-from models/mnist_cnn_v3.pt \
  --user-data-dir data/user_digits \
  --user-oversample 12
```

## Run The API

Start the backend locally:

```bash
uvicorn app.main:app --reload
```

Available endpoints:

- `GET /health`
- `POST /predict`

Example request:

```bash
curl -X POST -F "file=@digit.png" http://localhost:8000/predict
```

## Tests

Run backend tests:

```bash
pytest
```

## Docker

Build the image:

```bash
docker build -t digit-recognizer-backend .
```

Run the container:

```bash
docker run --rm -p 8000:8000 --env-file .env digit-recognizer-backend
```

## Notes

- The API loads the configured model once at startup.
- Startup fails fast if the model checkpoint is missing.
- For best real-world accuracy, fine-tune on your own handwriting samples.
