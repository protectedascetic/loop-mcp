"""
Loop MCP Server — remote HTTP (SSE transport via FastMCP)

Env vars required:
  DATABASE_URL       — same Postgres URL as the bot
  TELEGRAM_USER_ID   — your numeric Telegram user ID
  MCP_SECRET         — optional; token auth via ?token= in the URL
"""

import json
import logging
import os
from datetime import datetime, timedelta

from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
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

TELEGRAM_USER_ID = int(os.environ["TELEGRAM_USER_ID"])

# Journal types never appear in action-loop tools
_JOURNAL_TYPES = ("observation", "reflection", "note")

# ---------------------------------------------------------------------------
# Attention score (mirrors bot.py formula — kept in sync manually)
# ---------------------------------------------------------------------------

def _attention_score(
    priority: str, loop_type: str, created_at: datetime,
    snooze_count: int, note_count: int, total_events: int,
) -> float:
    importance = {"low": 1.0, "medium": 2.0, "high": 3.0, "critical": 4.0}.get(priority, 2.0)
    days_open = max((datetime.utcnow() - created_at).days, 0)
    age_weight = 1.0 + min(days_open / 10, 4.0)
    activity = 1.0 + (snooze_count * 0.6) + (note_count * 0.3) + (total_events * 0.1)
    type_mult = {
        "decision": 1.5, "concern": 1.4, "waiting": 1.2,
        "task": 1.0, "opportunity": 0.9, "idea": 0.7,
    }.get(loop_type, 1.0)
    return min(round(importance * age_weight * activity * type_mult * 5, 1), 200.0)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def _get_user_id(db: AsyncSession) -> int | None:
    r = await db.execute(
        text("SELECT id FROM users WHERE telegram_id = :tid"),
        {"tid": TELEGRAM_USER_ID},
    )
    row = r.first()
    return row.id if row else None


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("loop")


@mcp.tool()
async def get_open_loops(priority: str = "", type: str = "", limit: int = 50) -> str:
    """
    Get open loops from Loop. Optionally filter by priority (low/medium/high/critical)
    or type (task/waiting/decision/idea/concern/opportunity).
    Results sorted critical → high → medium → low.
    """
    async with SessionLocal() as db:
        uid = await _get_user_id(db)
        if not uid:
            return "User not found. Send /start to the Loop bot first."

        q = "SELECT id, title, type, priority, created_at, last_activity_at, recurrence FROM loops WHERE user_id = :uid AND status = 'open'"
        params: dict = {"uid": uid}

        if priority:
            q += " AND priority = :priority"
            params["priority"] = priority
        if type:
            q += " AND type = :type"
            params["type"] = type

        q += """
            ORDER BY CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
                     last_activity_at DESC
            LIMIT :lim
        """
        params["lim"] = min(int(limit), 100)

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


@mcp.tool()
async def search_loops(query: str, include_resolved: bool = False) -> str:
    """Search loops by keyword in title. Set include_resolved=true to search resolved loops too."""
    async with SessionLocal() as db:
        uid = await _get_user_id(db)
        if not uid:
            return "User not found."

        status_filter = "" if include_resolved else "AND status IN ('open', 'snoozed')"
        rows = (await db.execute(
            text(f"""
                SELECT id, title, type, priority, status, created_at
                FROM loops
                WHERE user_id = :uid AND LOWER(title) LIKE :q {status_filter}
                ORDER BY CASE status WHEN 'open' THEN 0 WHEN 'snoozed' THEN 1 ELSE 2 END,
                         CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END
                LIMIT 20
            """),
            {"uid": uid, "q": f"%{query.lower()}%"},
        )).fetchall()

        if not rows:
            return f"No loops found matching '{query}'."

        now = datetime.utcnow()
        lines = [f"{len(rows)} result(s) for '{query}':\n"]
        for r in rows:
            age = (now - r.created_at).days
            status_tag = f" [{r.status}]" if r.status != "open" else ""
            lines.append(f"[{r.id}] {r.title}  |  {r.priority} {r.type}{status_tag}  |  {age}d")
        return "\n".join(lines)


@mcp.tool()
async def get_brain_summary() -> str:
    """
    Cognitive dashboard: open loop counts, urgent items, stale items, top attention loads,
    pending decisions, and journal entry counts (observations/reflections/notes).
    Use this for a quick overall state-of-mind read before diving into specifics.
    """
    async with SessionLocal() as db:
        uid = await _get_user_id(db)
        if not uid:
            return "User not found."

        now = datetime.utcnow()
        stale_cutoff = now - timedelta(days=14)

        # Counts
        total_action = (await db.execute(
            text(f"SELECT COUNT(*) FROM loops WHERE user_id=:uid AND status='open' AND type NOT IN {_JOURNAL_TYPES}"),
            {"uid": uid},
        )).scalar()
        journal_count = (await db.execute(
            text(f"SELECT COUNT(*) FROM loops WHERE user_id=:uid AND status='open' AND type IN {_JOURNAL_TYPES}"),
            {"uid": uid},
        )).scalar()
        snoozed = (await db.execute(
            text("SELECT COUNT(*) FROM loops WHERE user_id=:uid AND status='snoozed'"), {"uid": uid}
        )).scalar()
        decisions = (await db.execute(
            text("SELECT COUNT(*) FROM loops WHERE user_id=:uid AND status='open' AND type='decision'"), {"uid": uid}
        )).scalar()
        waiting = (await db.execute(
            text("SELECT COUNT(*) FROM loops WHERE user_id=:uid AND status='open' AND type='waiting'"), {"uid": uid}
        )).scalar()

        urgent = (await db.execute(
            text(f"""SELECT id, title, type, priority FROM loops
                    WHERE user_id=:uid AND status='open' AND priority IN ('critical','high')
                    AND type NOT IN {_JOURNAL_TYPES}
                    ORDER BY CASE priority WHEN 'critical' THEN 0 ELSE 1 END LIMIT 10"""),
            {"uid": uid},
        )).fetchall()

        stale = (await db.execute(
            text(f"""SELECT id, title FROM loops
                    WHERE user_id=:uid AND status='open' AND last_activity_at < :cutoff
                    AND type NOT IN {_JOURNAL_TYPES} LIMIT 8"""),
            {"uid": uid, "cutoff": stale_cutoff},
        )).fetchall()

        # Top attention loads (action loops with event counts)
        action_rows = (await db.execute(
            text(f"""SELECT l.id, l.title, l.type, l.priority, l.created_at,
                        COUNT(e.id) FILTER (WHERE e.event_type='snoozed') as snooze_count,
                        COUNT(e.id) FILTER (WHERE e.event_type IN ('noted','updated')) as note_count,
                        COUNT(e.id) as total_events
                    FROM loops l
                    LEFT JOIN events e ON e.loop_id = l.id
                    WHERE l.user_id=:uid AND l.status='open' AND l.type NOT IN {_JOURNAL_TYPES}
                    GROUP BY l.id, l.title, l.type, l.priority, l.created_at"""),
            {"uid": uid},
        )).fetchall()

        scored = sorted(
            [
                (r, _attention_score(r.priority, r.type, r.created_at,
                                     r.snooze_count, r.note_count, r.total_events))
                for r in action_rows
            ],
            key=lambda x: -x[1],
        )

        lines = [
            f"Brain summary — {total_action} action loops · {snoozed} snoozed · "
            f"{decisions} decisions · {waiting} waiting · {journal_count} journal entries\n"
        ]

        if urgent:
            lines.append(f"⚡ URGENT ({len(urgent)}):")
            for r in urgent:
                lines.append(f"  [{r.id}] {r.title}  ({r.priority} {r.type})")

        if scored:
            lines.append(f"\n🎯 TOP ATTENTION LOADS:")
            for r, score in scored[:5]:
                snooze_note = f"  [snoozed {r.snooze_count}×]" if r.snooze_count >= 2 else ""
                lines.append(f"  [{r.id}] {r.title}  ({r.type}){snooze_note}")

        if stale:
            lines.append(f"\n🕸 STALE 14+ days ({len(stale)}):")
            for r in stale:
                lines.append(f"  [{r.id}] {r.title}")

        return "\n".join(lines)


@mcp.tool()
async def get_reading_list() -> str:
    """Get all saved reading list items (URLs saved to Loop)."""
    async with SessionLocal() as db:
        uid = await _get_user_id(db)
        if not uid:
            return "User not found."

        rows = (await db.execute(
            text("""SELECT id, title, loop_meta, created_at FROM loops
                    WHERE user_id=:uid AND status='open' AND loop_meta->>'is_reading' = 'true'
                    ORDER BY created_at DESC LIMIT 30"""),
            {"uid": uid},
        )).fetchall()

        if not rows:
            return "Reading list is empty."

        now = datetime.utcnow()
        lines = [f"{len(rows)} item(s):\n"]
        for r in rows:
            meta = r.loop_meta if isinstance(r.loop_meta, dict) else json.loads(r.loop_meta or "{}")
            url = meta.get("url", "")
            domain = meta.get("url_domain", "")
            age = (now - r.created_at).days
            lines.append(f"[{r.id}] {r.title}  |  {domain}  |  {age}d")
            if url:
                lines.append(f"       {url}")
        return "\n".join(lines)


@mcp.tool()
async def get_snoozed_loops() -> str:
    """Get all currently snoozed loops with their wake dates."""
    async with SessionLocal() as db:
        uid = await _get_user_id(db)
        if not uid:
            return "User not found."

        rows = (await db.execute(
            text("""SELECT id, title, type, priority, snoozed_until FROM loops
                    WHERE user_id=:uid AND status='snoozed' ORDER BY snoozed_until ASC"""),
            {"uid": uid},
        )).fetchall()

        if not rows:
            return "No snoozed loops."

        lines = [f"{len(rows)} snoozed:\n"]
        for r in rows:
            wake = r.snoozed_until.strftime("%a %b %-d") if r.snoozed_until else "unknown"
            lines.append(f"[{r.id}] {r.title}  |  {r.priority} {r.type}  |  wakes {wake}")
        return "\n".join(lines)


@mcp.tool()
async def add_loop(title: str, type: str = "task", priority: str = "medium", summary: str = "") -> str:
    """
    Capture a new loop. Types: task/waiting/decision/idea/concern/opportunity.
    Priorities: low/medium/high/critical.
    """
    async with SessionLocal() as db:
        uid = await _get_user_id(db)
        if not uid:
            return "User not found. Send /start to the Loop bot first."

        title = title.strip()[:500]
        if not title:
            return "Error: title is required."

        meta = json.dumps({"summary": summary, "source": "mcp"})
        result = await db.execute(
            text("""INSERT INTO loops (user_id, raw_text, title, type, priority, status, entities, loop_meta, created_at, updated_at, last_activity_at)
                    VALUES (:uid, :raw, :title, :type, :priority, 'open', '[]', :meta, NOW(), NOW(), NOW())
                    RETURNING id"""),
            {"uid": uid, "raw": title, "title": title, "type": type, "priority": priority, "meta": meta},
        )
        loop_id = result.scalar()
        await db.commit()
        return f"✅ Captured [{loop_id}]: {title}  ({priority} {type})"


@mcp.tool()
async def get_journal_entries(type: str = "", limit: int = 30) -> str:
    """
    Get journal entries: observations, reflections, and notes.
    Optionally filter by type: 'observation' | 'reflection' | 'note'.
    These are non-actionable captures — personal processing, patterns noticed, raw facts saved.
    Useful for understanding emotional context, recurring themes, and self-awareness patterns.
    """
    async with SessionLocal() as db:
        uid = await _get_user_id(db)
        if not uid:
            return "User not found."

        type_filter = ""
        params: dict = {"uid": uid, "lim": min(int(limit), 100)}
        if type and type in _JOURNAL_TYPES:
            type_filter = "AND type = :jtype"
            params["jtype"] = type
        else:
            type_filter = f"AND type IN {_JOURNAL_TYPES}"

        rows = (await db.execute(
            text(f"""SELECT id, title, type, created_at, loop_meta
                    FROM loops
                    WHERE user_id=:uid AND status='open' {type_filter}
                    ORDER BY created_at DESC LIMIT :lim"""),
            params,
        )).fetchall()

        if not rows:
            t = f" of type '{type}'" if type else ""
            return f"No journal entries{t}."

        now = datetime.utcnow()
        lines = [f"{len(rows)} journal entry/entries:\n"]
        for r in rows:
            age = (now - r.created_at).days
            age_str = f"{age}d" if age > 0 else "today"
            meta = r.loop_meta if isinstance(r.loop_meta, dict) else json.loads(r.loop_meta or "{}")
            summary = meta.get("summary", "")
            lines.append(f"[{r.id}] [{r.type}] {r.title}  |  {age_str}")
            if summary:
                lines.append(f"       {summary}")
        return "\n".join(lines)


@mcp.tool()
async def get_attention_scores(limit: int = 10) -> str:
    """
    Get open action loops ranked by attention score — a composite of priority, age,
    snooze history, and update frequency. High snooze count signals avoidance.
    Use this to understand what's consuming the most cognitive bandwidth,
    or to identify loops that may be stuck or being avoided.
    """
    async with SessionLocal() as db:
        uid = await _get_user_id(db)
        if not uid:
            return "User not found."

        rows = (await db.execute(
            text(f"""SELECT l.id, l.title, l.type, l.priority, l.created_at,
                        COUNT(e.id) FILTER (WHERE e.event_type='snoozed') as snooze_count,
                        COUNT(e.id) FILTER (WHERE e.event_type IN ('noted','updated')) as note_count,
                        COUNT(e.id) as total_events
                    FROM loops l
                    LEFT JOIN events e ON e.loop_id = l.id
                    WHERE l.user_id=:uid AND l.status='open' AND l.type NOT IN {_JOURNAL_TYPES}
                    GROUP BY l.id, l.title, l.type, l.priority, l.created_at"""),
            {"uid": uid},
        )).fetchall()

        if not rows:
            return "No open action loops."

        scored = sorted(
            [
                (r, _attention_score(r.priority, r.type, r.created_at,
                                     r.snooze_count, r.note_count, r.total_events))
                for r in rows
            ],
            key=lambda x: -x[1],
        )[:min(int(limit), 50)]

        now = datetime.utcnow()
        lines = [f"Top {len(scored)} loops by attention score:\n"]
        for i, (r, score) in enumerate(scored, 1):
            age = (now - r.created_at).days
            snooze_note = f"  [snoozed {r.snooze_count}×]" if r.snooze_count >= 2 else ""
            note_note = f"  [{r.note_count} updates]" if r.note_count else ""
            lines.append(
                f"{i}. [{r.id}] {r.title}  |  {r.priority} {r.type}  |  {age}d{snooze_note}{note_note}"
            )
        return "\n".join(lines)


@mcp.tool()
async def update_loop(loop_id: int, note: str) -> str:
    """
    Add a note or update to an existing open loop.
    Use when you want to record new information against a loop without resolving it —
    e.g. 'Finance said next week', 'Priya confirmed', 'Still blocked on this'.
    Updates last_activity_at so the loop surfaces as recently active.
    """
    async with SessionLocal() as db:
        uid = await _get_user_id(db)
        if not uid:
            return "User not found."

        note = note.strip()[:500]
        if not note:
            return "Error: note text is required."

        row = (await db.execute(
            text("SELECT id, title, loop_meta FROM loops WHERE id=:lid AND user_id=:uid AND status IN ('open','snoozed')"),
            {"lid": loop_id, "uid": uid},
        )).first()

        if not row:
            return f"Loop {loop_id} not found or not open."

        meta = row.loop_meta if isinstance(row.loop_meta, dict) else json.loads(row.loop_meta or "{}")
        notes = meta.get("notes", [])
        notes.append(note)
        meta["notes"] = notes

        await db.execute(
            text("""UPDATE loops SET
                        loop_meta=:meta,
                        last_activity_at=NOW(),
                        updated_at=NOW()
                    WHERE id=:lid AND user_id=:uid"""),
            {"meta": json.dumps(meta), "lid": loop_id, "uid": uid},
        )
        await db.execute(
            text("INSERT INTO events (loop_id, event_type, data, timestamp) VALUES (:lid, 'noted', :data, NOW())"),
            {"lid": loop_id, "data": json.dumps({"note": note, "source": "mcp"})},
        )
        await db.commit()
        return f"📝 Updated [{loop_id}]: {row.title}\n   → {note}"


@mcp.tool()
async def resolve_loop(loop_id: int) -> str:
    """Mark a loop as resolved. Use the loop ID from get_open_loops."""
    async with SessionLocal() as db:
        uid = await _get_user_id(db)
        if not uid:
            return "User not found."

        result = await db.execute(
            text("""UPDATE loops SET status='resolved', updated_at=NOW()
                    WHERE id=:lid AND user_id=:uid AND status='open' RETURNING title"""),
            {"lid": loop_id, "uid": uid},
        )
        row = result.first()
        await db.commit()

        if not row:
            return f"Loop {loop_id} not found or not open."
        return f"✅ Resolved: {row.title}"


# ---------------------------------------------------------------------------
# ASGI app — plain Starlette, no TrustedHostMiddleware (Railway-compatible)
# ---------------------------------------------------------------------------

_sse = SseServerTransport("/messages/")


async def _handle_sse(request: Request) -> None:
    async with _sse.connect_sse(
        request.scope, request.receive, request._send
    ) as (recv_stream, send_stream):
        await mcp._mcp_server.run(
            recv_stream,
            send_stream,
            mcp._mcp_server.create_initialization_options(),
        )


async def _health(request: Request) -> Response:
    return Response("ok", media_type="text/plain")


app = Starlette(
    routes=[
        Route("/health", endpoint=_health),
        Route("/sse", endpoint=_handle_sse),
        Mount("/messages/", app=_sse.handle_post_message),
    ]
)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
