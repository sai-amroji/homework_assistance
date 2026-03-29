from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os

# =========================
# IMPORT YOUR ROUTERS
# =========================
from routes.auth_routes import router as auth_router
from routes.ask_routes import router as ask_router
from routes.image_routes import router as image_router
from routes.grammar_routes import router as grammar_router

# =========================
# LIFESPAN
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    from services.ai_service import model as ai_model
    from services.grammar_service import grammar_model
    print("✅ AI models loaded and ready.")
    yield
    print("🛑 Shutting down.")

# =========================
# APP
# =========================
app = FastAPI(lifespan=lifespan)

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# =========================
# ✅ SERVE FULL FRONTEND (FINAL FIX)
# =========================
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

# =========================
# INCLUDE API ROUTES
# =========================
app.include_router(auth_router)
app.include_router(ask_router)
app.include_router(image_router)
app.include_router(grammar_router)

# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}