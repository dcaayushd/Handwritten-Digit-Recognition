# Handwritten Digit Recognition Backend

Documentation for the Python API and training pipeline. This file intentionally excludes Flutter setup and frontend usage.

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
├── README_BACKEND.md
└── requirements.txt
```

## Local Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the environment file:

```bash
cp .env.example .env
```

## Environment Variables

Example values:

```env
MODEL_VERSION=v3
MODEL_PATH=models/mnist_cnn_v3.pt
LOG_LEVEL=INFO
MIN_PREDICTION_CONFIDENCE=0.55
MIN_PREDICTION_MARGIN=0.18
```

## Train the Model

Train the current production checkpoint:

```bash
python -m app.train --version v3 --architecture mnist_resnet --epochs 14 --batch-size 128 --learning-rate 0.0015
```

This downloads MNIST into `data/`, trains the model, evaluates it, and saves the checkpoint in `models/`.

## Fine-Tune on Your Own Handwriting

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

Then fine-tune from the existing model:

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

## Run the API Locally

Start the backend:

```bash
uvicorn app.main:app --reload
```

The API serves:

- `GET /health`
- `POST /predict`

Example request:

```bash
curl -X POST -F "file=@digit.png" http://localhost:8000/predict
```

## Run Tests

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
- If the model is missing, startup fails fast.
- Real phone-photo handwriting can differ from plain MNIST. Fine-tuning on personal samples improves accuracy.
