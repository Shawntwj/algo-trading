"""Dev server entrypoint.

Run with either:
    python -m api.main
    uvicorn api.app:app --reload
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python -m api.main` from the project root regardless of cwd.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn


def main() -> None:
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
