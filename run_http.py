"""Run the pg-mcp MCP server over Streamable HTTP for browser-based testing.

Not part of the shipped server (stdio is the production transport); this is a
local test harness so the MCP endpoint can be reached from a browser page or
the MCP Inspector at http://127.0.0.1:8000/mcp .

CORS is wide open on purpose — this harness exists for local demos only.
"""

import anyio
import uvicorn
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from pg_mcp.server import mcp


def create_app():
    """Build the MCP Starlette app wrapped in permissive CORS middleware."""
    app = mcp.streamable_http_app()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Mcp-Session-Id", "Mcp-Protocol-Version"],
    )
    return app


def main() -> None:
    uvicorn.run(create_app(), host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
