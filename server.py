"""
Loop MCP Server — remote HTTP (SSE transport)

Exposes your Loop Postgres database as MCP tools so Claude can query,
capture, and resolve loops directly from any Claude interface.

Env vars required:
  DATABASE_URL       — same Postgres URL as the bot
  MCP_SECRET         — a secret string; clients must send as Bearer token
  TELEGRAM_USER_ID   — your Telegram user ID (find via @userinfobot)

Deploy on Railway as a separate service in the same project.
Set these in Railway → Variables for this service.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp import types
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_raw_db_url = os.environ["DATABASE_URL"]
DATABASE_URL = (
    _raw_db_url
    .replace("postgresql://", "postgresql+asyncpg://", 1)
    .replace("postgres://", "postgresql+asyncpg://", 1)
)

MCP_SECRET = os.environ.get("MCP_SECRET", "")
TELEGRAM_USER_ID = int(os.environ["TELEGRAM_USER_ID"])

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = Server("loop")


@mcp.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_open_loops",
            description=(
                "Get Mayank's open loops from Loop. "
                "Optionally filter by priority (low/medium/high/critical) "
                "or type (task/waiting/decision/idea/concern/opportunity). "
                "Results are sorted: critical → high → medium → low."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Only return loops of this priority",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["task", "waiting", "decision", "idea", "concern", "opportunity"],
                        "description": "Only return loops of this type",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "description": "Max number of loops to return (max 100)",
                    },
                },
            },
        ),
        types.Tool(
            name="search_loops",
            description="Search loops by keyword in title. Returns open and resolved loops.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keyword"},
                    "include_resolved": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include resolved loops in results",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get_brain_summary",
            description=(
                "Get a structured cognitive dashboard: total count, urgent items "
                "(critical + high), stale items (14+ days inactive), snoozed count, "
                "pending decisions. Best for a quick status check."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="get_reading_list",
            description="Get all saved reading list items (links saved to Loop via URL).",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="get_snoozed_loops",
            description="Get loops that are currently snoozed, with their wake dates.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="add_loop",
            description=(
                "Add a new loop/task to capture into Loop. "
                "Use this when the user asks to add, capture, or save something. "
                "Infer type and priority from context if not specified."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Concise loop title, max 60 chars",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["task", "waiting", "decision", "idea", "concern", "opportunity"],
                        "default": "task",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "default": "medium",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Optional one-sentence context note",
                    },
                },
                "required": ["title"],
            },
        ),
        types.Tool(
            name="resolve_loop",
            description="Mark a loop as resolved/done. Use the loop ID from get_open_loops.",
            inputSchema={
                "type": "object",
                "properties": {
                    "loop_id": {"type": "integer", "description": "The loop's numeric ID"},
                },
                "required": ["loop_id"],
            },
        ),
    ]


@mcp.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        async with SessionLocal() as db:
            result = await _dispatch(db, name, arguments)
        return [types.TextContent(type="text", text=result)]
    except Exception as e:
        logger.exception("Tool %s failed: %s", name, e)
        return [types.TextContent(type="text", text=f"Error: {e}")]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def _get_user_id(db: AsyncSession) -> int | None:
    r = await db.execute(
        text("SELECT id FROM users WHERE telegram_id = :tid"),
        {"tid": TELEGRAM_USER_ID},
    )
    row = r.first()
    return row.id if row else None


async def _dispatch(db: AsyncSession, name: str, args: dict) -> str:
    if name == "get_open_loops":
        return await _get_open_loops(db, args)
    elif name == "search_loops":
        return await _search_loops(db, args)
    elif name == "get_brain_summary":
        return await _get_brain_summary(db)
    elif name == "get_reading_list":
        return await _get_reading_list(db)
    elif name == "get_snoozed_loops":
        return await _get_snoozed_loops(db)
    elif name == "add_loop":
        return await _add_loop(db, args)
    elif name == "resolve_loop":
        return await _resolve_loop(db, args)
    return f"Unknown tool: {name}"


async def _get_open_loops(db: AsyncSession, args: dict) -> str:
    uid = await _get_user_id(db)
    if not uid:
        return "User not found. Send /start to the Loop bot first."

    q = """
        SELECT id, title, type, priority, created_at, last_activity_at, recurrence
        FROM loops
        WHERE user_id = :uid AND status = 'open'
    """
    params: dict = {"uid": uid}

    if args.get("priority"):
        q += " AND priority = :priority"
        params["priority"] = args["priority"]
    if args.get("type"):
        q += " AND type = :type"
        params["type"] = args["type"]

    q += """
        ORDER BY
          CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
          last_activity_at DESC
        LIMIT :lim
    """
    params["lim"] = min(int(args.get("limit", 50)), 100)

    rows = (await db.execute(text(q), params)).fetchall()
    if not rows:
        return "No open loops found."

    now = datetime.utcnow()
    lines = [f"{len(rows)} open loop(s):\n"]
    for r in rows:
        age = (now - r.created_at).days
        age_str = f"{age}d" if age > 0 else "today"
        rec = f"  [recurring: {r.recurrence}]" if r.recurrence else ""
        lines.append(f"[{r.id}] {r.title}  |  {r.priority} {r.type}  |  {age_str}{rec}")

    return "\n".join(lines)


async def _search_loops(db: AsyncSession, args: dict) -> str:
    uid = await _get_user_id(db)
    if not uid:
        return "User not found."

    include_resolved = args.get("include_resolved", False)
    status_filter = "" if include_resolved else " AND status IN ('open', 'snoozed')"

    rows = (await db.execute(
        text(f"""
            SELECT id, title, type, priority, status, created_at
            FROM loops
            WHERE user_id = :uid AND LOWER(title) LIKE :q {status_filter}
            ORDER BY CASE status WHEN 'open' THEN 0 WHEN 'snoozed' THEN 1 ELSE 2 END,
                     CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END
            LIMIT 20
        """),
        {"uid": uid, "q": f"%{args.get('query', '').lower()}%"},
    )).fetchall()

    if not rows:
        return f"No loops found matching '{args.get('query')}'."

    now = datetime.utcnow()
    lines = [f"{len(rows)} result(s) for '{args.get('query')}':\n"]
    for r in rows:
        age = (now - r.created_at).days
        status_tag = f" [{r.status}]" if r.status != "open" else ""
        lines.append(f"[{r.id}] {r.title}  |  {r.priority} {r.type}{status_tag}  |  {age}d")

    return "\n".join(lines)


async def _get_brain_summary(db: AsyncSession) -> str:
    uid = await _get_user_id(db)
    if not uid:
        return "User not found."

    stale_cutoff = datetime.utcnow() - timedelta(days=14)

    total = (await db.execute(
        text("SELECT COUNT(*) FROM loops WHERE user_id=:uid AND status='open'"), {"uid": uid}
    )).scalar()

    snoozed = (await db.execute(
        text("SELECT COUNT(*) FROM loops WHERE user_id=:uid AND status='snoozed'"), {"uid": uid}
    )).scalar()

    urgent_rows = (await db.execute(
        text("""
            SELECT id, title, type, priority FROM loops
            WHERE user_id=:uid AND status='open' AND priority IN ('critical','high')
            ORDER BY CASE priority WHEN 'critical' THEN 0 ELSE 1 END
            LIMIT 10
        """),
        {"uid": uid},
    )).fetchall()

    stale_rows = (await db.execute(
        text("""
            SELECT id, title FROM loops
            WHERE user_id=:uid AND status='open' AND last_activity_at < :cutoff
            LIMIT 8
        """),
        {"uid": uid, "cutoff": stale_cutoff},
    )).fetchall()

    decisions = (await db.execute(
        text("SELECT COUNT(*) FROM loops WHERE user_id=:uid AND status='open' AND type='decision'"),
        {"uid": uid},
    )).scalar()

    waiting = (await db.execute(
        text("SELECT COUNT(*) FROM loops WHERE user_id=:uid AND status='open' AND type='waiting'"),
        {"uid": uid},
    )).scalar()

    lines = [
        f"Brain summary — {total} open · {snoozed} snoozed · {decisions} decisions · {waiting} waiting\n"
    ]

    if urgent_rows:
        lines.append(f"⚡ URGENT ({len(urgent_rows)}):")
        for r in urgent_rows:
            lines.append(f"  [{r.id}] {r.title}  ({r.priority} {r.type})")

    if stale_rows:
        lines.append(f"\n🕸 STALE 14+ days ({len(stale_rows)}):")
        for r in stale_rows:
            lines.append(f"  [{r.id}] {r.title}")

    return "\n".join(lines)


async def _get_reading_list(db: AsyncSession) -> str:
    uid = await _get_user_id(db)
    if not uid:
        return "User not found."

    rows = (await db.execute(
        text("""
            SELECT id, title, loop_meta, created_at FROM loops
            WHERE user_id=:uid AND status='open' AND loop_meta->>'is_reading' = 'true'
            ORDER BY created_at DESC LIMIT 30
        """),
        {"uid": uid},
    )).fetchall()

    if not rows:
        return "Reading list is empty. Send any URL to the Loop bot to save it."

    now = datetime.utcnow()
    lines = [f"{len(rows)} item(s) in reading list:\n"]
    for r in rows:
        meta = r.loop_meta if isinstance(r.loop_meta, dict) else json.loads(r.loop_meta or "{}")
        url = meta.get("url", "")
        domain = meta.get("url_domain", "")
        age = (now - r.created_at).days
        age_str = f"{age}d" if age > 0 else "today"
        lines.append(f"[{r.id}] {r.title}  |  {domain or 'link'}  |  {age_str}")
        if url:
            lines.append(f"       {url}")

    return "\n".join(lines)


async def _get_snoozed_loops(db: AsyncSession) -> str:
    uid = await _get_user_id(db)
    if not uid:
        return "User not found."

    rows = (await db.execute(
        text("""
            SELECT id, title, type, priority, snoozed_until FROM loops
            WHERE user_id=:uid AND status='snoozed'
            ORDER BY snoozed_until ASC
        """),
        {"uid": uid},
    )).fetchall()

    if not rows:
        return "No snoozed loops."

    lines = [f"{len(rows)} snoozed loop(s):\n"]
    for r in rows:
        wake = r.snoozed_until.strftime("%a %b %-d") if r.snoozed_until else "unknown"
        lines.append(f"[{r.id}] {r.title}  |  {r.priority} {r.type}  |  wakes {wake}")

    return "\n".join(lines)


async def _add_loop(db: AsyncSession, args: dict) -> str:
    uid = await _get_user_id(db)
    if not uid:
        return "User not found. Send /start to the Loop bot first."

    title = (args.get("title") or "").strip()[:500]
    if not title:
        return "Error: title is required."

    loop_type = args.get("type", "task")
    priority = args.get("priority", "medium")
    summary = args.get("summary", "")
    meta = json.dumps({"summary": summary, "source": "mcp"})

    result = await db.execute(
        text("""
            INSERT INTO loops
              (user_id, raw_text, title, type, priority, status, entities, loop_meta,
               created_at, updated_at, last_activity_at)
            VALUES
              (:uid, :raw, :title, :type, :priority, 'open', '[]', :meta,
               NOW(), NOW(), NOW())
            RETURNING id
        """),
        {
            "uid": uid, "raw": title, "title": title,
            "type": loop_type, "priority": priority, "meta": meta,
        },
    )
    loop_id = result.scalar()
    await db.commit()

    return f"✅ Captured [{loop_id}]: {title}  ({priority} {loop_type})"


async def _resolve_loop(db: AsyncSession, args: dict) -> str:
    uid = await _get_user_id(db)
    if not uid:
        return "User not found."

    loop_id = args.get("loop_id")
    if not loop_id:
        return "Error: loop_id is required."

    result = await db.execute(
        text("""
            UPDATE loops SET status='resolved', updated_at=NOW()
            WHERE id=:lid AND user_id=:uid AND status='open'
            RETURNING title
        """),
        {"lid": loop_id, "uid": uid},
    )
    row = result.first()
    await db.commit()

    if not row:
        return f"Loop {loop_id} not found or not open."

    return f"✅ Resolved: {row.title}"


# ---------------------------------------------------------------------------
# SSE transport + Starlette app
# ---------------------------------------------------------------------------

sse = SseServerTransport("/messages/")


def _check_auth(request: Request) -> bool:
    if not MCP_SECRET:
        return True  # no secret configured — open (not recommended in prod)
    auth_header = request.headers.get("Authorization", "")
    token_param = request.query_params.get("token", "")
    return auth_header == f"Bearer {MCP_SECRET}" or token_param == MCP_SECRET


async def handle_sse(request: Request) -> Response:
    if not _check_auth(request):
        return Response("Unauthorized", status_code=401)
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await mcp.run(streams[0], streams[1], mcp.create_initialization_options())
    return Response()


async def handle_messages(request: Request) -> Response:
    await sse.handle_post_message(request.scope, request.receive, request._send)
    return Response()


async def handle_health(request: Request) -> Response:
    return Response("ok")


app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=handle_messages),
        Route("/health", endpoint=handle_health),
    ]
)
