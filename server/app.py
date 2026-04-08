"""
FastAPI application for the CustomerSupportEnv Environment.

Endpoints:
    POST /reset   — Start a new episode
    POST /step    — Submit an action for the current complaint
    GET  /state   — Get current environment state
    GET  /schema  — Action / Observation schemas
    WS   /ws      — WebSocket persistent session (low-latency)
    GET  /health  — Healthcheck
    GET  /web     — Interactive web dashboard
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required.  Install with: uv sync"
    ) from e

try:
    from models import SupportAction, SupportObservation
    from server.customer_support_environment import CustomerSupportEnvironment
except (ImportError, ModuleNotFoundError):
    try:
        from ..models import SupportAction, SupportObservation
        from .customer_support_environment import CustomerSupportEnvironment
    except (ImportError, ModuleNotFoundError):
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models import SupportAction, SupportObservation
        from server.customer_support_environment import CustomerSupportEnvironment

app = create_app(
    CustomerSupportEnvironment,
    SupportAction,
    SupportObservation,
    env_name="customer_support_env",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
