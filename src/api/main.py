from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Project-internal imports
# ---------------------------------------------------------------------------
from shared.logger import get_logger
from src.api.routes.classical_routes import router as classical_router
from src.api.routes.ingest_routes import router as ingest_router
from src.api.routes.quantum_routes import router as quantum_router
from src.api.routes.eval_routes import router as eval_router
from src.api.routes.hybrid_routes import router as hybrid_router
from src.api.routes.documents_routes import router as documents_router

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_V1_PREFIX: str = "/api/v1"

APP_TITLE: str = "HQC Topic Modeling API"

APP_DESCRIPTION: str = (
    "Hybrid Quantum–Classical Optimization Method for "
    "Topic Modeling and Document Clustering."
)

APP_VERSION: str = "1.0.0"

ALLOWED_ORIGINS: list[str] = ["*"]


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info(
        "HQC API starting | version=%s | prefix=%s",
        APP_VERSION,
        API_V1_PREFIX,
    )

    yield

    logger.info("HQC API shutting down")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:

    application = FastAPI(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    application.include_router(
    hybrid_router,
    prefix=f"{API_V1_PREFIX}/hybrid",
    tags=["Hybrid"],
)
    
    application.include_router(
    documents_router,
    prefix=f"{API_V1_PREFIX}/documents",
    tags=["Documents"],
)

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    application.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Request middleware
    # ------------------------------------------------------------------
    @application.middleware("http")
    async def request_context_middleware(
        request: Request,
        call_next: Any,
    ) -> Any:

        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.perf_counter()

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"

        logger.info(
            "request | id=%s | method=%s | path=%s | status=%d | elapsed_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )

        return response

    # ------------------------------------------------------------------
    # Exception handlers
    # ------------------------------------------------------------------
    @application.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:

        request_id = getattr(request.state, "request_id", "unknown")

        logger.exception(
            "Unhandled exception | request_id=%s | path=%s",
            request_id,
            request.url.path,
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_server_error",
                "detail": "An unexpected error occurred.",
                "code": 500,
            },
        )

    @application.exception_handler(ValueError)
    async def value_error_handler(
        request: Request,
        exc: ValueError,
    ) -> JSONResponse:

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "detail": str(exc),
                "code": 422,
            },
        )

    @application.exception_handler(FileNotFoundError)
    async def not_found_error_handler(
        request: Request,
        exc: FileNotFoundError,
    ) -> JSONResponse:

        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": "resource_not_found",
                "detail": str(exc),
                "code": 404,
            },
        )

    # ------------------------------------------------------------------
    # Router registration
    # ------------------------------------------------------------------
    application.include_router(
        ingest_router,
        prefix=f"{API_V1_PREFIX}/ingest",
        tags=["Ingestion"],
    )

    application.include_router(
        classical_router,
        prefix=f"{API_V1_PREFIX}/classical",
        tags=["Classical NLP"],
    )

    application.include_router(
        quantum_router,
        prefix=f"{API_V1_PREFIX}/quantum",
        tags=["Quantum"],
    )

    application.include_router(
        eval_router,
        prefix=f"{API_V1_PREFIX}/eval",
        tags=["Evaluation"],
    )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    @application.get("/api/v1/health", tags=["Meta"])
    async def health_check() -> dict[str, str]:

        return {
            "status": "ok",
            "version": APP_VERSION,
        }

    # ------------------------------------------------------------------
    # Root endpoint
    # ------------------------------------------------------------------
    @application.get("/")
    async def root() -> dict[str, str]:

        return {
            "message": "Hybrid Quantum-Classical Topic Modeling API Running"
        }

    # ------------------------------------------------------------------
    # API root
    # ------------------------------------------------------------------
    @application.get(f"{API_V1_PREFIX}", tags=["Meta"])
    async def api_root() -> dict[str, Any]:

        return {
            "title": APP_TITLE,
            "version": APP_VERSION,
            "routes": [
                f"{API_V1_PREFIX}/ingest",
                f"{API_V1_PREFIX}/classical",
                f"{API_V1_PREFIX}/quantum",
                f"{API_V1_PREFIX}/eval",
                f"{API_V1_PREFIX}/documents",
                f"{API_V1_PREFIX}/hybrid",
            ],
        }

    logger.info(
        "FastAPI app created | routers=ingest,classical,quantum,eval"
    )

    return application


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app: FastAPI = create_app()
