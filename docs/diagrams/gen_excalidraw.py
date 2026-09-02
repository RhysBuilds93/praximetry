"""Generate Excalidraw scene files for praximetry architecture diagrams.

Grounded in the code on branch `rich-cli` (HEAD 2237f5b): symbol names, routes,
and control flow all match src/praximetry/ as of this commit.
"""

from __future__ import annotations

import json
import random
import time

random.seed(1729)
NOW = int(time.time() * 1000)


def _rid() -> str:
    return "".join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(16))


def _seed() -> int:
    return random.randint(1, 2**31)


PALETTE = {
    "oss": ("#2f9e44", "#ebfbee"),
    "cloud": ("#e8590c", "#fff0e6"),
    "store": ("#1971c2", "#e7f5ff"),
    "signal": ("#f08c00", "#fff9db"),
    "plain": ("#1e1e1e", "#ffffff"),
    "muted": ("#868e96", "#f1f3f5"),
}


class Scene:
    def __init__(self) -> None:
        self.elements: list[dict] = []

    def _base(self, **kw) -> dict:
        d = dict(
            id=_rid(),
            x=0,
            y=0,
            width=0,
            height=0,
            angle=0,
            strokeColor="#1e1e1e",
            backgroundColor="transparent",
            fillStyle="solid",
            strokeWidth=2,
            strokeStyle="solid",
            roughness=1,
            opacity=100,
            groupIds=[],
            frameId=None,
            roundness=None,
            seed=_seed(),
            version=1,
            versionNonce=_seed(),
            isDeleted=False,
            boundElements=[],
            updated=NOW,
            link=None,
            locked=False,
        )
        d.update(kw)
        return d

    def region(self, x, y, w, h, label) -> str:
        r = self._base(
            type="rectangle",
            x=x,
            y=y,
            width=w,
            height=h,
            strokeColor=PALETTE["muted"][0],
            backgroundColor="transparent",
            strokeStyle="dashed",
            strokeWidth=1.5,
            roughness=0,
            roundness={"type": 3},
        )
        self.elements.append(r)
        t = self._base(
            type="text",
            x=x + 16,
            y=y + 12,
            width=w - 32,
            height=24,
            text=label,
            originalText=label,
            fontSize=18,
            fontFamily=2,
            textAlign="left",
            verticalAlign="top",
            containerId=None,
            lineHeight=1.25,
            baseline=18,
            autoResize=True,
            strokeColor=PALETTE["muted"][0],
        )
        self.elements.append(t)
        return r["id"]

    def node(self, x, y, w, h, text, kind="plain", *, dashed=False, font=15) -> str:
        stroke, bg = PALETTE[kind]
        rect = self._base(
            type="rectangle",
            x=x,
            y=y,
            width=w,
            height=h,
            strokeColor=stroke,
            backgroundColor=bg,
            fillStyle="solid",
            strokeStyle="dashed" if dashed else "solid",
            roughness=1,
            roundness={"type": 3},
        )
        tid = _rid()
        rect["boundElements"] = [{"type": "text", "id": tid}]
        self.elements.append(rect)
        nlines = text.count("\n") + 1
        t = self._base(
            id=tid,
            type="text",
            x=x + 8,
            y=y + h / 2 - nlines * font * 0.625,
            width=w - 16,
            height=nlines * font * 1.25,
            text=text,
            originalText=text,
            fontSize=font,
            fontFamily=2,
            textAlign="center",
            verticalAlign="middle",
            containerId=rect["id"],
            lineHeight=1.25,
            baseline=font,
            autoResize=True,
            strokeColor=stroke,
        )
        self.elements.append(t)
        return rect["id"]

    def label(self, x, y, text, *, color="#1e1e1e", font=13, align="left") -> str:
        nlines = text.count("\n") + 1
        t = self._base(
            type="text",
            x=x,
            y=y,
            width=max(len(line) for line in text.split("\n")) * font * 0.58,
            height=nlines * font * 1.25,
            text=text,
            originalText=text,
            fontSize=font,
            fontFamily=2,
            textAlign=align,
            verticalAlign="top",
            containerId=None,
            lineHeight=1.25,
            baseline=font,
            autoResize=True,
            strokeColor=color,
        )
        self.elements.append(t)
        return t["id"]

    def _elem(self, eid: str) -> dict:
        return next(e for e in self.elements if e["id"] == eid)

    def arrow(
        self,
        a: str,
        b: str,
        *,
        label: str | None = None,
        dashed=False,
        color="#1e1e1e",
        start_side=None,
        end_side=None,
    ) -> str:
        ea, eb = self._elem(a), self._elem(b)

        def anchor(e, side):
            cx, cy = e["x"] + e["width"] / 2, e["y"] + e["height"] / 2
            if side == "r":
                return e["x"] + e["width"], cy
            if side == "l":
                return e["x"], cy
            if side == "t":
                return cx, e["y"]
            if side == "b":
                return cx, e["y"] + e["height"]
            return cx, cy

        if start_side is None or end_side is None:
            dx = (eb["x"] + eb["width"] / 2) - (ea["x"] + ea["width"] / 2)
            dy = (eb["y"] + eb["height"] / 2) - (ea["y"] + ea["height"] / 2)
            if abs(dx) > abs(dy):
                start_side = start_side or ("r" if dx > 0 else "l")
                end_side = end_side or ("l" if dx > 0 else "r")
            else:
                start_side = start_side or ("b" if dy > 0 else "t")
                end_side = end_side or ("t" if dy > 0 else "b")

        x1, y1 = anchor(ea, start_side)
        x2, y2 = anchor(eb, end_side)
        arr = self._base(
            type="arrow",
            x=x1,
            y=y1,
            width=abs(x2 - x1),
            height=abs(y2 - y1),
            points=[[0, 0], [x2 - x1, y2 - y1]],
            strokeColor=color,
            strokeWidth=2,
            roughness=1,
            strokeStyle="dashed" if dashed else "solid",
            roundness={"type": 2},
            elbowed=False,
            startArrowhead=None,
            endArrowhead="arrow",
            lastCommittedPoint=None,
            startBinding={"elementId": a, "focus": 0, "gap": 6, "fixedPoint": None},
            endBinding={"elementId": b, "focus": 0, "gap": 6, "fixedPoint": None},
        )
        self.elements.append(arr)
        ea.setdefault("boundElements", []).append({"type": "arrow", "id": arr["id"]})
        eb.setdefault("boundElements", []).append({"type": "arrow", "id": arr["id"]})
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            self.label(mx - len(label) * 3, my - 16, label, color=color, font=12, align="center")
        return arr["id"]

    def dump(self, path: str) -> None:
        doc = {
            "type": "excalidraw",
            "version": 2,
            "source": "praximetry/scratchpad/gen_excalidraw.py",
            "elements": self.elements,
            "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None},
            "files": {},
        }
        with open(path, "w") as f:
            json.dump(doc, f, indent=2)
        print("wrote", path, f"({len(self.elements)} elements)")


# ---------------------------------------------------------------------------
# Diagram 1 — system overview: OSS package vs praximetry-cloud
# ---------------------------------------------------------------------------


def diagram_system(path: str) -> None:
    s = Scene()
    s.label(40, 24, "praximetry — end to end", font=26, color="#1e1e1e")
    s.label(
        40,
        58,
        "OSS package runs entirely in the customer's process. praximetry-cloud never imports agent code.\n"
        "Dependency is one-way: praximetry-cloud → praximetry.  Verified against branch rich-cli.",
        font=13,
        color="#495057",
    )

    s.region(30, 100, 560, 620, "Open source · praximetry  (your process)")
    s.region(650, 100, 520, 620, "Closed source · praximetry-cloud")

    agent = s.node(70, 150, 220, 60, "your agent code\n(unchanged)", "plain")
    init = s.node(
        70,
        250,
        220,
        66,
        "px.init()\nlazily patches openai / anthropic /\nlitellm / gemini SDKs",
        "oss",
    )
    stg = s.node(
        330,
        250,
        230,
        66,
        '@px.stage("…")   [optional]\nnames a pipeline step;\nregisters it in STAGE_REGISTRY',
        "oss",
    )
    call = s.node(
        70, 360, 220, 54, "patched create() / acreate()\ninstrument.patch._instrument", "oss"
    )
    rec = s.node(70, 452, 220, 54, "runtime.record_call()\ngate: get_config().enabled", "oss")
    store = s.node(
        60,
        548,
        240,
        60,
        "store.save_call()\nSQLite  .praximetry/praximetry.db\n(WAL, thread-local)",
        "store",
    )
    summ = s.node(340, 548, 230, 60, "praximetry summary\nper-stage cost / token table", "oss")

    csync = s.node(
        330,
        360,
        230,
        66,
        "cloud_sync.enqueue()   [if API key]\nbackground worker,\ndrop-on-full, never blocks",
        "oss",
        dashed=True,
    )

    evalc = s.node(
        330,
        452,
        230,
        62,
        "praximetry eval / optimize\ncapture request shape, raise\nbefore the real LLM call",
        "oss",
    )
    applyn = s.node(70, 640, 220, 56, "praximetry apply\nwrites .praximetry/overrides.json", "oss")

    api = s.node(
        690,
        160,
        440,
        58,
        "POST /api/eval/captures  ·  /api/optimize/captures  ·  /api/traces\nbearer auth: PRAXIMETRY_API_KEY",
        "cloud",
        font=13,
    )
    score = s.node(
        690,
        260,
        440,
        58,
        "cloud_eval.score_captures\nreal model call (Bedrock) + AI judge",
        "cloud",
    )
    pg = s.node(690, 360, 440, 54, "Postgres — tenant-scoped, row-level security", "cloud")
    dash = s.node(690, 452, 440, 58, "Dashboard\nObserve · Evaluate · Optimise", "cloud")
    winner = s.node(
        690, 552, 440, 58, "GET /api/optimize/winner\nwinning policy from a completed run", "cloud"
    )

    s.arrow(agent, init, end_side="t")
    s.arrow(init, call, label="every call")
    s.arrow(stg, call, label="stage ctx")
    s.arrow(call, rec)
    s.arrow(rec, store)
    s.arrow(store, summ, label="reads")
    s.arrow(rec, csync, start_side="r", end_side="l", dashed=True)
    s.arrow(csync, api, dashed=True, label="traces")
    s.arrow(stg, evalc, start_side="b", end_side="t", label="re-run by name")
    s.arrow(evalc, api, label="captured shapes")
    s.arrow(api, score, end_side="t")
    s.arrow(score, pg)
    s.arrow(pg, dash)
    s.arrow(dash, winner, label="optimise run")
    s.arrow(
        winner,
        applyn,
        start_side="l",
        end_side="r",
        dashed=True,
        color="#e8590c",
        label="fetch winner",
    )
    s.arrow(
        applyn,
        call,
        start_side="t",
        end_side="l",
        dashed=True,
        color="#e8590c",
        label="overrides honoured in-flight\n(_apply_overrides)",
    )

    s.dump(path)


# ---------------------------------------------------------------------------
# Diagram 2 — the recording path (how a call becomes a stored Call)
# ---------------------------------------------------------------------------


def diagram_recording(path: str) -> None:
    s = Scene()
    s.label(40, 24, "The recording path", font=26)
    s.label(
        40,
        58,
        "Three producers converge on the store. Only the SDK-patch and manual paths run through record_call\n"
        "today — the OTel path writes straight to save_call (PRA-64). Branch rich-cli.",
        font=13,
        color="#495057",
    )

    p1 = s.node(
        60, 120, 240, 50, "patched SDK client\nopenai / anthropic / litellm / gemini", "oss"
    )
    p2 = s.node(60, 210, 240, 50, "manual: px.record_call(...)\n(providers with no patcher)", "oss")
    p3 = s.node(60, 300, 240, 50, "OpenTelemetry span\ninstrument_otel() / record_spans()", "muted")

    prep = s.node(
        360,
        110,
        250,
        70,
        "_prepare_call()\n· _apply_overrides(kwargs)\n  model / prompt_transform",
        "oss",
    )
    hook = s.node(
        360,
        210,
        250,
        60,
        "capture_hook set?\nfire pre-flight hook → raise\n(eval / optimize capture)",
        "signal",
    )
    real = s.node(360, 300, 250, 50, "original(**kwargs)\nreal provider API call", "plain")
    parse = s.node(
        360,
        380,
        250,
        60,
        "adapter.parse_response()\nstreaming: wrap.py records\non stream exhaustion",
        "oss",
    )

    mapspan = s.node(
        360, 470, 250, 50, "otel.map_span(name, attrs)\n→ builds Call directly", "muted"
    )

    recc = s.node(
        700,
        250,
        260,
        84,
        "runtime.record_call(**kw)\n· if not config.enabled: return\n· stage = current_stage()\n· parent_call_id = _current_call",
        "store",
    )
    save = s.node(720, 380, 220, 46, "store.save_call(call)\nSQLite", "store")
    enq = s.node(
        720,
        470,
        220,
        60,
        "cloud_sync.note_run()\ncloud_sync.enqueue(call)\nif cloud_sync.is_running()",
        "store",
        dashed=True,
    )

    s.arrow(p1, prep)
    s.arrow(prep, hook)
    s.arrow(hook, real, label="not capturing")
    s.arrow(real, parse)
    s.arrow(parse, recc, label="_record(...)")
    s.arrow(p2, recc, start_side="r", end_side="l")
    s.arrow(p3, mapspan, start_side="b", end_side="l")
    s.arrow(
        mapspan,
        save,
        start_side="r",
        end_side="l",
        dashed=True,
        color="#868e96",
        label="bypasses record_call",
    )
    s.arrow(recc, save)
    s.arrow(recc, enq, dashed=True)

    s.dump(path)


# ---------------------------------------------------------------------------
# Diagram 3 — eval / optimize capture flow
# ---------------------------------------------------------------------------


def diagram_capture(path: str) -> None:
    s = Scene()
    s.label(40, 24, "praximetry eval — capture without calling an LLM", font=24)
    s.label(
        40,
        56,
        "capture_request() runs the real stage code up to its first outbound-LLM-shaped call, then unwinds.\n"
        "No network contact, no Call persisted. src/praximetry/eval/capture.py, branch rich-cli.",
        font=13,
        color="#495057",
    )

    c1 = s.node(60, 110, 300, 46, "praximetry eval --stage X -m my_agent", "oss")
    c2 = s.node(60, 180, 300, 50, "client_from_env()\nPRAXIMETRY_API_KEY / _API_URL", "oss")
    c3 = s.node(60, 254, 300, 50, "import my_agent\n@px.stage fns → STAGE_REGISTRY", "oss")
    c4 = s.node(60, 328, 300, 46, "client.fetch_corpus()  →  GET /api/eval/corpus", "cloud")

    s.region(430, 96, 470, 366, "for each Example:  capture_request(ex)")
    d1 = s.node(
        455, 140, 420, 44, "fn = STAGE_REGISTRY[ex.stage]   (else CaptureError)", "oss", font=13
    )
    d2 = s.node(
        455,
        200,
        420,
        46,
        "runtime.record_call = _intercept  (monkeypatch)\nwith patch.capturing(_hook):",
        "signal",
        font=13,
    )
    d3 = s.node(
        455,
        268,
        420,
        46,
        "call_stage(fn, ex)\nreal retrieval / templating fills dynamic values",
        "oss",
        font=13,
    )
    d4 = s.node(
        455,
        334,
        420,
        46,
        "first LLM-shaped call → raise _CaptureSignal\nunwind before original(...) runs",
        "signal",
        font=13,
    )
    d5 = s.node(
        455, 400, 420, 44, "CapturedRequest(provider, model, messages, tools)", "store", font=13
    )

    e1 = s.node(430, 500, 470, 46, "client.push_captures()  →  POST /api/eval/captures", "cloud")
    e2 = s.node(
        430, 566, 470, 50, "cloud scores synchronously\n→ quality · pass_rate · cost_usd", "cloud"
    )
    e3 = s.node(
        430,
        636,
        470,
        46,
        "exit 0 pass   ·   1 below --fail-under   ·   2 gate couldn't run",
        "plain",
    )

    s.arrow(c1, c2)
    s.arrow(c2, c3)
    s.arrow(c3, c4)
    s.arrow(c4, d1, start_side="r", end_side="l")
    s.arrow(d1, d2)
    s.arrow(d2, d3)
    s.arrow(d3, d4)
    s.arrow(d4, d5)
    s.arrow(d5, e1, start_side="b", end_side="t")
    s.arrow(e1, e2)
    s.arrow(e2, e3)

    s.dump(path)


if __name__ == "__main__":
    import os

    base = os.path.dirname(os.path.abspath(__file__))
    diagram_system(f"{base}/01-system-oss-cloud.excalidraw")
    diagram_recording(f"{base}/02-recording-path.excalidraw")
    diagram_capture(f"{base}/03-eval-capture-flow.excalidraw")
