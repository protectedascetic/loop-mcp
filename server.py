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
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

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
    Get a cognitive dashboard: total open loops, urgent items (critical+high),
    stale items (14+ days inactive), snoozed count, pending decisions.
    """
    async with SessionLocal() as db:
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
        decisions = (await db.execute(
            text("SELECT COUNT(*) FROM loops WHERE user_id=:uid AND status='open' AND type='decision'"), {"uid": uid}
        )).scalar()
        waiting = (await db.execute(
            text("SELECT COUNT(*) FROM loops WHERE user_id=:uid AND status='open' AND type='waiting'"), {"uid": uid}
        )).scalar()

        urgent = (await db.execute(
            text("""SELECT id, title, type, priority FROM loops
                    WHERE user_id=:uid AND status='open' AND priority IN ('critical','high')
                    ORDER BY CASE priority WHEN 'critical' THEN 0 ELSE 1 END LIMIT 10"""),
            {"uid": uid},
        )).fetchall()

        stale = (await db.execute(
            text("SELECT id, title FROM loops WHERE user_id=:uid AND status='open' AND last_activity_at < :cutoff LIMIT 8"),
            {"uid": uid, "cutoff": stale_cutoff},
        )).fetchall()

        lines = [f"Brain summary — {total} open · {snoozed} snoozed · {decisions} decisions · {waiting} waiting\n"]
        if urgent:
            lines.append(f"⚡ URGENT ({len(urgent)}):")
            for r in urgent:
                lines.append(f"  [{r.id}] {r.title}  ({r.priority} {r.type})")
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
# Run
# ---------------------------------------------------------------------------

# Expose the Starlette SSE app for uvicorn (used by Procfile)
app = mcp.sse_app()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
