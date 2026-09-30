import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from app import auth
from app.bus import bus
from app.console import nl_parser, query_runner
from app.db import init_db, recent_decisions
from app.enforcement.mock_api import router as device_router
from app.enforcement.pep_middleware import PEPMiddleware
from app.enforcement.pipeline import log_and_publish_resolution
from app.enforcement.review_queue import review_queue
from app.ingestion.pcap_replay import replay as pcap_replay
from app.orchestrator.llm_worker import LLMTriage
from app.orchestrator.rules import get_ruleset
from app.traffic.generator import run as traffic_run
from app import resources as protected_resources

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    protected_resources.init_if_missing()
    get_ruleset()  # seed the rule set

    # Ingestion source: PCAP if configured, synthetic traffic otherwise.
    manual_only = os.environ.get("ZTF_MANUAL_ONLY", "").strip().lower() in ("1", "true", "yes")
    pcap_path = os.environ.get("ZTF_PCAP_PATH")
    if manual_only:
     ingest_task = None
     print("[main] ingestion: disabled — manual/API requests only")
    elif pcap_path:
        speed = float(os.environ.get("ZTF_PCAP_SPEED", "10.0"))
        min_interval = float(os.environ.get("ZTF_MIN_EVENT_INTERVAL", "0"))
        ingest_task = asyncio.create_task(pcap_replay(pcap_path, speed=speed, min_event_interval=min_interval))
        print(f"[main] ingestion: PCAP replay ({pcap_path}, speed={speed}x,min_interval={min_interval}s)")
    else:
        ingest_task = asyncio.create_task(traffic_run())
        print("[main] ingestion: synthetic traffic generator")

    # LLM policy worker — always runs (uses heuristic if no API key).
    triage_task = asyncio.create_task(LLMTriage().run())

    # Sweeper for pending reviews that hit their timeout → auto-deny.
    sweeper_task = asyncio.create_task(_review_sweeper())

    try:
        yield
    finally:
        if ingest_task:
         ingest_task.cancel()
         triage_task.cancel()
         sweeper_task.cancel()


async def _review_sweeper() -> None:
    """Every 10s, auto-deny pending reviews past their expiry."""
    while True:
        await asyncio.sleep(10.0)
        try:
            expired = review_queue.sweep_expired()
            for item in expired:
                await log_and_publish_resolution(item)
                await bus.publish({"event_type": "review_resolved", "id": item.id})
        except Exception as e:
            print(f"[review_sweeper] error: {e}")


app = FastAPI(title="Zero-Trust Framework", lifespan=lifespan)
app.add_middleware(PEPMiddleware)
templates = Jinja2Templates(directory=BASE_DIR / "templates")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.include_router(device_router)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")


@app.get("/simulator", response_class=HTMLResponse)
async def simulator(request: Request):
    """Access Terminal — a real client that requests protected resources.

    Requires an authenticated session. Redirects to /login if none.
    """
    token = request.cookies.get("ztf_session")
    if auth.get_session(token) is None:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(request, "simulator.html")


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Sign-in / sign-up page. If already logged in, jump to the terminal."""
    token = request.cookies.get("ztf_session")
    if auth.get_session(token) is not None:
        return RedirectResponse("/simulator", status_code=302)
    return templates.TemplateResponse(request, "login.html")


@app.post("/api/signup")
async def api_signup(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "expected JSON body"}, status_code=400)
    ok, result = auth.signup(str(body.get("username", "")), str(body.get("password", "")))
    if not ok:
        return JSONResponse({"error": result}, status_code=400)
    resp = JSONResponse({"ok": True, "username": body["username"]})
    resp.set_cookie(
        "ztf_session", result,
        max_age=auth.SESSION_TTL_SECONDS,
        httponly=True, samesite="strict", path="/",
    )
    return resp


@app.post("/api/login")
async def api_login(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "expected JSON body"}, status_code=400)
    ok, result = auth.login(str(body.get("username", "")), str(body.get("password", "")))
    if not ok:
        return JSONResponse({"error": result}, status_code=401)
    resp = JSONResponse({"ok": True, "username": body["username"]})
    resp.set_cookie(
        "ztf_session", result,
        max_age=auth.SESSION_TTL_SECONDS,
        httponly=True, samesite="strict", path="/",
    )
    return resp


@app.post("/api/logout")
async def api_logout(request: Request):
    token = request.cookies.get("ztf_session")
    auth.logout(token)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("ztf_session", path="/")
    return resp


@app.get("/api/session")
async def api_session(request: Request):
    """Return the current session's username, or 401 if not logged in."""
    token = request.cookies.get("ztf_session")
    username = auth.get_session(token)
    if username is None:
        return JSONResponse({"authenticated": False}, status_code=401)
    return {"authenticated": True, "username": username}


@app.get("/api/whoami")
async def whoami(request: Request):
    """Reveal the identity the framework will attribute to this caller.

    Same rule as PEP middleware: X-Device-ID header wins if present,
    else the client's network address. Also reports the logged-in user
    if any.
    """
    header_id = request.headers.get("X-Device-ID")
    ip = request.client.host if request.client else None
    token = request.cookies.get("ztf_session")
    username = auth.get_session(token)
    return {
        "identity": header_id or ip or "unknown",
        "source": "header" if header_id else "client_ip",
        "username": username,
       # "note": "The framework identifies your device by its network address. "
        #        "Every request you send is judged against this identity's trust score.",
    }


@app.get("/api/devices")
async def devices():
    """List the known device_ids from device_config.csv."""
    return {"devices": protected_resources.list_devices()}


@app.get("/events")
async def events(request: Request):
    """SSE stream. Emits 'decision' events (table rows) and 'rule_update'
    events (policy panel cards). HTMX consumers listen for the event
    name they care about.
    """
    async def stream():
        queue = bus.subscribe()
        try:
            while True:
                if await request.is_disconnected():
                    break
                event = await queue.get()
                et = event.get("event_type", "")
                if et == "rule_update":
                    yield {"event": "rule_update", "data": render_rule_update(event)}
                elif et in ("review_pending", "review_updated", "review_resolved"):
                    # No HTML payload — the dashboard's pending panel refetches
                    # /api/review/panel whenever any of these three events fires.
                    yield {"event": et, "data": event.get("id", "")}
                else:
                    yield {"event": "decision", "data": render_row(event)}
        finally:
            bus.unsubscribe(queue)

    return EventSourceResponse(stream())


@app.get("/api/audit")
async def audit(limit: int = 50):
    return JSONResponse(await recent_decisions(limit=min(limit, 500)))


@app.get("/api/rules")
async def rules():
    """Current rule set snapshot: blocklist entries, threshold overrides,
    recent policy changes with LLM reasoning.
    """
    return JSONResponse(get_ruleset().snapshot())


# ---------------------------------------------------------------------------
# Human review endpoints
# ---------------------------------------------------------------------------

@app.get("/api/review/panel", response_class=HTMLResponse)
async def review_panel():
    """Render the current pending-review panel as an HTML fragment."""
    return HTMLResponse(_render_review_panel())


@app.post("/api/review/{pid}/allow", response_class=HTMLResponse)
async def review_allow(request: Request, pid: str):
    item = review_queue.resolve(pid, "allowed", by="human")
    if item is None:
        return HTMLResponse(_render_review_panel(), status_code=404)
    user = auth.get_session(request.cookies.get("ztf_session"))
    await log_and_publish_resolution(item, resolved_by_user=user)
    await bus.publish({"event_type": "review_resolved", "id": pid})
    return HTMLResponse(_render_review_panel())


@app.post("/api/review/{pid}/deny", response_class=HTMLResponse)
async def review_deny(request: Request, pid: str):
    item = review_queue.resolve(pid, "denied", by="human")
    if item is None:
        return HTMLResponse(_render_review_panel(), status_code=404)
    user = auth.get_session(request.cookies.get("ztf_session"))
    await log_and_publish_resolution(item, resolved_by_user=user)
    await bus.publish({"event_type": "review_resolved", "id": pid})
    return HTMLResponse(_render_review_panel())


@app.post("/api/review/{pid}/retry", response_class=HTMLResponse)
async def review_retry(pid: str, request: Request):
    form = await request.form()
    try:
        retry_after = max(5, int(form.get("retry_after_seconds", 60)))
    except (ValueError, TypeError):
        retry_after = 60
    item = review_queue.resolve(pid, "retry", retry_after_seconds=retry_after, by="human")
    if item is None:
        return HTMLResponse(_render_review_panel(), status_code=404)
    user = auth.get_session(request.cookies.get("ztf_session"))
    await log_and_publish_resolution(item, resolved_by_user=user)
    await bus.publish({"event_type": "review_resolved", "id": pid})
    return HTMLResponse(_render_review_panel())


def _render_review_panel() -> str:
    import html as html_mod

    def esc(v):
        return html_mod.escape(str(v))

    items = review_queue.pending()
    if not items:
        return (
            '<div class="p-8 text-center text-sm text-gray-400">'
            'No pending reviews. Escalates will appear here for human decision.'
            '</div>'
        )

    parts = []
    for i in items:
        rec = (i.llm_recommendation or "").lower()
        rec_class = {
            "allow": "bg-green-100 text-green-800 border-green-300",
            "deny": "bg-red-100 text-red-800 border-red-300",
            "retry": "bg-amber-100 text-amber-800 border-amber-300",
        }.get(rec, "bg-gray-100 text-gray-600 border-gray-300")
        rec_label = rec.upper() if rec else "advising…"
        conf = (
            f" · conf {i.llm_confidence:.2f}"
            if i.llm_confidence is not None else ""
        )
        reasoning_html = (
            f'<div class="mt-2 p-2 border {rec_class} rounded text-xs">'
            f'<div class="font-medium mb-1">LLM: {esc(rec_label)}{esc(conf)}</div>'
            f'<div class="italic">{esc(i.llm_reasoning or "(waiting for advisor)")}</div>'
            f'</div>'
        )
        expires = int(max(0, i.expires_at - __import__("time").time()))
        parts.append(
            f'<div class="border-b border-gray-200 last:border-b-0 p-3">'
            f'<div class="flex justify-between items-baseline">'
            f'<div class="text-sm font-medium">{esc(i.src)} \u2192 <span class="font-mono">{esc(i.action)}</span></div>'
            f'<div class="text-xs text-gray-500 font-mono">{esc(i.id)} \u00b7 {expires}s</div>'
            f'</div>'
            f'<div class="text-xs text-gray-600 mt-1">PDP score {i.pdp_score:.2f} \u00b7 {esc(i.pdp_reason)[:90]}</div>'
            f'{reasoning_html}'
            f'<div class="flex gap-2 mt-2">'
            f'<button hx-post="/api/review/{esc(i.id)}/allow" hx-target="#pending-reviews-container" hx-swap="innerHTML" '
            f'  class="px-3 py-1 bg-green-600 hover:bg-green-700 text-white text-xs rounded font-medium">Allow</button>'
            f'<button hx-post="/api/review/{esc(i.id)}/deny" hx-target="#pending-reviews-container" hx-swap="innerHTML" '
            f'  class="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-xs rounded font-medium">Deny</button>'
            f'<form hx-post="/api/review/{esc(i.id)}/retry" hx-target="#pending-reviews-container" hx-swap="innerHTML" '
            f'  class="flex gap-1 items-center">'
            f'<select name="retry_after_seconds" class="text-xs border border-gray-300 rounded px-1 py-0.5">'
            f'<option value="30">30s</option><option value="60" selected>1m</option>'
            f'<option value="300">5m</option><option value="3600">1h</option>'
            f'</select>'
            f'<button type="submit" class="px-3 py-1 bg-amber-600 hover:bg-amber-700 text-white text-xs rounded font-medium">Retry</button>'
            f'</form>'
            f'</div>'
            f'</div>'
        )
    return "".join(parts)


@app.post("/api/console", response_class=HTMLResponse)
async def console(request: Request):
    """Natural-language security console.

    Body: form-encoded 'q' from HTMX submit.
    Returns an HTML fragment: user query bubble + parsed operation +
    result table, ready to append to the chat panel.
    """
    form = await request.form()
    query = str(form.get("q", "")).strip()
    if not query:
        return HTMLResponse("")

    parsed = await nl_parser.parse(query)
    operation = parsed.get("operation", "")
    args = parsed.get("args", {}) or {}
    explanation = parsed.get("explanation", "")

    result = await query_runner.run(operation, args)
    return HTMLResponse(_render_console_exchange(query, operation, args, explanation, result))


def _render_console_exchange(query: str, operation: str, args: dict, explanation: str, result: dict) -> str:
    import html as html_mod

    def esc(v):
        return html_mod.escape(str(v))

    columns = result.get("columns") or []
    rows = result.get("rows") or []
    meta = result.get("meta") or {}

    if columns and rows:
        header = "".join(
            f'<th class="text-left px-2 py-1 text-xs font-medium text-gray-600 uppercase tracking-wide">{esc(c)}</th>'
            for c in columns
        )
        body_rows = "".join(
            "<tr class='border-t border-gray-200'>" + "".join(
                f'<td class="px-2 py-1 text-sm font-mono align-top">{esc(r.get(c, ""))}</td>'
                for c in columns
            ) + "</tr>"
            for r in rows
        )
        table = (
            f'<div class="overflow-x-auto mt-2"><table class="w-full">'
            f'<thead class="bg-gray-50"><tr>{header}</tr></thead>'
            f'<tbody>{body_rows}</tbody></table></div>'
        )
    elif "error" in meta:
        table = f'<div class="text-sm text-red-700 mt-2">{esc(meta["error"])}</div>'
    else:
        table = '<div class="text-sm text-gray-500 mt-2 italic">No matching rows.</div>'

    meta_bits = []
    if "count" in meta:
        meta_bits.append(f"{meta['count']} row(s)")
    if "rule_version" in meta:
        meta_bits.append(f"rule set v{meta['rule_version']}")
    meta_str = " \u00b7 ".join(meta_bits)

    args_str = ", ".join(f"{k}={v}" for k, v in args.items() if v is not None) or "none"
    meta_suffix = f" \u00b7 {esc(meta_str)}" if meta_str else ""
    explanation_html = (
        f'<div class="text-sm text-gray-700 italic">{esc(explanation)}</div>'
        if explanation else ""
    )

    return (
        # user bubble
        f'<div class="flex justify-end mb-3">'
        f'<div class="bg-blue-100 text-blue-900 rounded-lg px-3 py-2 max-w-lg text-sm">{esc(query)}</div>'
        f'</div>'
        # system bubble
        f'<div class="mb-4">'
        f'<div class="bg-white border border-gray-200 rounded-lg px-3 py-2">'
        f'<div class="text-xs text-gray-500 mb-1">'
        f'<span class="font-mono">{esc(operation)}</span> \u00b7 args: <span class="font-mono">{esc(args_str)}</span>'
        f'{meta_suffix}'
        f'</div>'
        f'{explanation_html}'
        f'{table}'
        f'</div>'
        f'</div>'
    )


def render_row(event: dict) -> str:
    verdict_class = {
        "allow": "text-green-800 bg-green-100",
        "escalate": "text-amber-800 bg-amber-100",
        "deny": "text-red-800 bg-red-100",
    }.get(event["verdict"], "text-gray-800 bg-gray-100")

    score = event.get("score")
    score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "\u2014"

    return (
        f'<tr class="border-b border-gray-200">'
        f'<td class="px-3 py-2 text-sm text-gray-600 font-mono">{event["ts"]}</td>'
        f'<td class="px-3 py-2 text-sm font-mono">{event["src"]}</td>'
        f'<td class="px-3 py-2 text-sm">{event["action"]}</td>'
        f'<td class="px-3 py-2 text-sm font-mono text-gray-700">{score_str}</td>'
        f'<td class="px-3 py-2 text-sm">'
        f'<span class="px-2 py-0.5 rounded {verdict_class} font-medium">{event["verdict"]}</span>'
        f'</td>'
        f'<td class="px-3 py-2 text-sm text-gray-600">{event["reason"]}</td>'
        f'</tr>'
    )


def render_rule_update(event: dict) -> str:
    kind = event.get("kind", "?")
    target = event.get("target", "?")
    version = event.get("version", "?")
    source = event.get("source", "?")
    confidence = float(event.get("confidence", 0.0))
    reasoning = str(event.get("llm_reasoning", ""))[:200]
    ts = event.get("ts", "")

    border_color = {
        "add_blocklist": "border-red-500",
        "remove_blocklist": "border-green-500",
        "override_thresholds": "border-blue-500",
    }.get(kind, "border-gray-500")

    detail = event.get("detail", {})
    detail_str = ", ".join(f"{k}={v}" for k, v in detail.items())
    reasoning_html = (
        f'<div class="text-xs text-gray-700 mt-1 italic">{reasoning}</div>'
        if reasoning else ""
    )

    return (
        f'<div class="border-l-4 {border_color} bg-white border border-gray-200 rounded-r px-3 py-2 mb-2">'
        f'<div class="flex justify-between items-center text-xs text-gray-500">'
        f'<span class="font-mono">v{version} \u00b7 {source} \u00b7 conf {confidence:.2f}</span>'
        f'<span class="font-mono">{ts}</span>'
        f'</div>'
        f'<div class="text-sm font-medium mt-1">{kind}: <span class="font-mono">{target}</span></div>'
        f'<div class="text-xs text-gray-600 mt-0.5">{detail_str}</div>'
        f'{reasoning_html}'
        f'</div>'
    )
