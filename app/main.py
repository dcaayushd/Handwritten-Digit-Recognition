from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.predict import ModelLoadError, Predictor
from app.utils import InputValidationError, UnsupportedFileTypeError, setup_logging

logger = logging.getLogger("digit_recognizer.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("Starting API with model paths=%s", [str(path) for path in settings.model_paths])
    app.state.settings = settings
    app.state.predictor = Predictor.from_settings(settings)
    yield


app = FastAPI(
    title="Handwritten Digit Recognizer API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_predictor(request: Request) -> Predictor:
    return request.app.state.predictor


def get_runtime_settings(request: Request) -> Settings:
    return request.app.state.settings


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    client_host = request.client.host if request.client else "unknown"
    logger.info("Incoming request method=%s path=%s client=%s", request.method, request.url.path, client_host)

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            "Unhandled error method=%s path=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Completed request method=%s path=%s status=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.exception_handler(UnsupportedFileTypeError)
async def unsupported_file_handler(_: Request, exc: UnsupportedFileTypeError) -> JSONResponse:
    logger.warning("Unsupported upload rejected: %s", exc)
    return JSONResponse(status_code=415, content={"detail": str(exc)})


@app.exception_handler(InputValidationError)
async def validation_error_handler(_: Request, exc: InputValidationError) -> JSONResponse:
    logger.warning("Invalid upload rejected: %s", exc)
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ModelLoadError)
async def model_error_handler(_: Request, exc: ModelLoadError) -> JSONResponse:
    logger.exception("Runtime model error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Model is unavailable."})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)) -> dict[str, float | int]:
    predictor = get_predictor(request)
    settings = get_runtime_settings(request)
    file_bytes = await file.read()
    prediction = predictor.predict_image_bytes(
        file_bytes,
        filename=file.filename,
        content_type=file.content_type,
    )
    logger.info(
        "Prediction response filename=%s digit=%s confidence=%.4f model=%s",
        file.filename,
        prediction.digit,
        prediction.confidence,
        ",".join(path.name for path in settings.model_paths),
    )
    return prediction.to_response()
