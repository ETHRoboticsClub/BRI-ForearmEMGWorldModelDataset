from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from femg.hand import HandMap, available_hands
from tools.explorer.data import Store

FRONTEND = Path(__file__).resolve().parent / "front" / "dist"


def create_app(data: Path, hand_port: int | None, host: str = "127.0.0.1") -> FastAPI:
    store = Store(data)
    hand = None
    if hand_port:
        from femg.hand import HandScene

        hand = HandScene(hand_port, host=host)

    app = FastAPI(title="femg explorer")
    app.add_middleware(GZipMiddleware, minimum_size=2048, compresslevel=4)

    def session(key: str):
        if key not in store.folders:
            raise HTTPException(404, f"no session {key}")
        return store.session(key)

    @app.get("/api/config")
    def config():
        return {"hand_port": hand_port if hand else None, "data": str(data)}

    @app.get("/api/hands")
    def hands():
        current = hand.hand.name if hand else None
        return {"current": current, "hands": [{"name": n, "title": t} for n, t in available_hands().items()]}

    @app.get("/api/hands/{name}")
    def hand_map(name: str):
        if name not in available_hands():
            raise HTTPException(404, f"no hand {name}")
        rules = HandMap(name).rules
        return [
            {
                "joint": r.joint,
                "label": r.label,
                "at_0": r.at_0,
                "at_1": r.at_1,
                "neutral_mean": r.neutral_mean,
                "source": r.source,
                "ratio": r.ratio,
                "fallback": r.fallback,
            }
            for r in rules
        ]

    @app.get("/api/sessions")
    def sessions():
        return store.summaries()

    @app.get("/api/sessions/{key:path}/overview")
    def overview(key: str, points: int = Query(1200, ge=50, le=4000)):
        return session(key).overview(points)

    @app.get("/api/sessions/{key:path}/window")
    def window(
        key: str,
        start: float,
        end: float,
        points: int = Query(1500, ge=50, le=8000),
        emg_scale: str = Query("uv", pattern="^(uv|active)$"),
        kin_scale: str = Query("rad", pattern="^(rad|unit)$"),
        channels: str = Query("ring8", pattern="^(ring8|all)$"),
    ):
        return session(key).window(start, end, points, emg_scale, kin_scale, channels)

    @app.get("/api/sessions/{key:path}")
    def details(key: str):
        return session(key).details()

    @app.websocket("/api/hand")
    async def hand_socket(socket: WebSocket):
        await socket.accept()
        try:
            while True:
                message = await socket.receive_json()
                if hand is None:
                    continue
                if "dark" in message:
                    hand.set_theme(message["dark"])
                if "labels" in message:
                    hand.show_labels(message["labels"])
                if message.get("hand") in available_hands() and message["hand"] != hand.hand.name:
                    hand.switch(message["hand"])
                if message.get("key") in store.folders:
                    s = store.session(message["key"])
                    frame = min(max(int(message["t"] * s.label_fs), 0), len(s.kin) - 1)
                    unit = message.get("unit", True)
                    hand.show((s.kin_unit if unit else s.kin)[frame], unit, s.kin_center)
        except WebSocketDisconnect:
            pass

    if FRONTEND.exists():
        app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
    return app
