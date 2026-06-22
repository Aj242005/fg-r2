"""
FastAPI Application — Traffic Violation Detection Server
==========================================================
Main entry point for the API server.
Start with: uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent))
from server.database import init_db
import config
from pipeline import ViolationPipeline
from server.routes import router, set_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup, cleanup on shutdown."""
    print("\n" + "=" * 60)
    print("🚦 TRAFFIC VIOLATION DETECTION SERVER")
    print("=" * 60)

    # Ensure output directory exists
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    # Ensure models directory exists
    config.MODELS_DIR.mkdir(exist_ok=True)

    # Initialize pipeline with available models
    try:
        pipeline = ViolationPipeline(
            violation_model_path=None,  # auto-detect
            plate_model_path=None,      # auto-detect
            enable_plate_reader=True,
        )
        set_pipeline(pipeline)
        print("\n✅ Server ready! Models loaded successfully.")
    except FileNotFoundError as e:
        print(f"\n⚠️  Warning: {e}")
        print("Server starting without some models.")
        print("Place model files in the 'models/' directory and restart.")

        # Try with just violation model
        try:
            pipeline = ViolationPipeline(
                violation_model_path=None,
                plate_model_path=None,
                enable_plate_reader=False,
            )
            set_pipeline(pipeline)
            print("✅ Server ready with violation detection only (no plate OCR).")
        except FileNotFoundError as e2:
            print(f"❌ No models available: {e2}")
            print("Please train models using the scripts in training/ and place .pt files in models/")

    # Initialize database tables
    try:
        await init_db()
        print("✅ Database tables initialized.")
    except Exception as e:
        print(f"⚠️  Database init warning: {e}")
        print("   Dashboard data features will be unavailable.")

    print(f"\n📡 Server running at http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    print(f"📖 API docs at http://localhost:{config.SERVER_PORT}/docs")
    print("=" * 60 + "\n")

    yield  # Server is running

    # Shutdown
    print("\n🛑 Shutting down server...")
    try:
        pipeline = set_pipeline(None)
    except Exception:
        pass


# ── Create FastAPI app ──
app = FastAPI(
    title="Traffic Violation Detection API",
    description=(
        "Dual-model traffic violation detection system. "
        "Detects helmet violations, tripling, red light jumping, illegal parking, "
        "stop line violations, and modified vehicles. "
        "Reads number plates via OCR. Both models run in parallel for speed."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow all for development) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount output directory for serving annotated files ──
config.OUTPUT_DIR.mkdir(exist_ok=True)
app.mount("/static/output", StaticFiles(directory=str(config.OUTPUT_DIR)), name="output_files")

# ── Include routes ──
app.include_router(router, prefix="/api")


# ── Root endpoint ──
@app.get("/", tags=["System"])
async def root():
    return {
        "service": "Traffic Violation Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/health",
            "analyze_image": "POST /api/analyze/image",
            "analyze_video": "POST /api/analyze/video",
            "analyze_stream": "WS /api/analyze/stream",
            "analyze_dataset": "POST /api/analyze/dataset",
        },
    }


# ── Run directly ──
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.app:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=True,
        log_level="info",
    )
