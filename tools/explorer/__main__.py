import argparse
import shutil
import subprocess
from pathlib import Path

import uvicorn

from tools.explorer.app import FRONTEND, create_app

FRONT = FRONTEND.parent


def outdated() -> bool:
    built = FRONTEND / "index.html"
    if not built.exists():
        return True
    sources = (
        [p for p in (FRONT / "src").rglob("*") if p.is_file()] + list(FRONT.glob("*.json")) + [FRONT / "index.html"]
    )
    return max(p.stat().st_mtime for p in sources) > built.stat().st_mtime


def build_frontend() -> None:
    npm = shutil.which("npm")
    if npm is None:
        raise SystemExit("npm not found: install Node.js to build the dashboard (tools/explorer/front)")
    if not (FRONT / "node_modules").exists():
        subprocess.run([npm, "install", "--no-fund", "--no-audit"], cwd=FRONT, check=True)
    subprocess.run([npm, "run", "build"], cwd=FRONT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tools.explorer", description="Dashboard to explore data/processed."
    )
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to open it from other machines")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--hand-port", type=int, default=8081, help="viser server of the hand")
    parser.add_argument("--no-hand", action="store_true")
    parser.add_argument("--no-build", action="store_true", help="serve the API only (for `npm run dev`)")
    args = parser.parse_args()

    if not args.no_build and outdated():
        build_frontend()
    app = create_app(args.data, None if args.no_hand else args.hand_port, host=args.host)
    print(f"femg explorer on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
