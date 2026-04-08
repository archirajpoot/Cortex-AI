"""
run.py — Single-port launcher for CustomerSupportEnv
======================================================
Combines the FastAPI backend + Gradio dashboard on ONE port.

    Dashboard : http://127.0.0.1:7860/
    API Docs  : http://127.0.0.1:7860/docs
    Health    : http://127.0.0.1:7860/health

Usage:
    .venv\\Scripts\\python.exe run.py
"""

import os
import sys

# ── ensure project root is importable ────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PORT     = int(os.getenv("PORT", "7860"))
HOST     = os.getenv("HOST", "127.0.0.1")
BASE_URL = f"http://127.0.0.1:{PORT}"

# ── 1. Build the FastAPI backend ──────────────────────────────────────────────
from models import SupportAction, SupportObservation                   # noqa: E402
from server.customer_support_environment import CustomerSupportEnvironment  # noqa: E402

from openenv.core.env_server.http_server import create_fastapi_app     # noqa: E402
from fastapi.responses import RedirectResponse                          # noqa: E402

fastapi_app = create_fastapi_app(
    CustomerSupportEnvironment,
    SupportAction,
    SupportObservation,
    max_concurrent_envs=4,
)

# ── 2. Redirect "/" → the Gradio UI ──────────────────────────────────────────
@fastapi_app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse("/ui/")

# ── 2.5 Health Check Endpoint ────────────────────────────────────────────────
@fastapi_app.get("/health", include_in_schema=False)
def health_endpoint():
    return {"status": "healthy"}


# ── 3. Import the Gradio module and patch its API URL ────────────────────────
#    The api_reset / api_step callbacks look up `app.API` at call-time,
#    so patching the module attribute here is sufficient.
import app as gradio_module                                             # noqa: E402
gradio_module.API = BASE_URL

# ── 4. Mount the Gradio Blocks app on the FastAPI app at /ui ─────────────────
import gradio as gr                                                     # noqa: E402

gr.mount_gradio_app(
    fastapi_app,
    gradio_module.demo,
    path="/ui",
)

# ── 5. Start everything ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    print()
    print("=" * 60)
    print("  🚀  CustomerSupportEnv — Single-Port Server")
    print("=" * 60)
    print(f"  Dashboard : {BASE_URL}/")
    print(f"  API Docs  : {BASE_URL}/docs")
    print(f"  Health    : {BASE_URL}/health")
    print("=" * 60)
    print()

    uvicorn.run(
        fastapi_app,
        host=HOST,
        port=PORT,
        log_level="info",
    )
