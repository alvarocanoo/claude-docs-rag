"""Drop and recreate the documents table — destructive, use with care."""

from __future__ import annotations

import asyncio
import sys

from claude_docs_rag.storage.db import connect
from claude_docs_rag.storage.schema import SCHEMA_SQL
from claude_docs_rag.storage.vector_store import count_documents

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def main() -> None:
    async with connect(register_vector=False) as conn, conn.cursor() as cur:
        await cur.execute("DROP TABLE IF EXISTS documents CASCADE")
        await cur.execute(SCHEMA_SQL)
        await conn.commit()
    n = await count_documents()
    print(f"Reset complete. Row count: {n}")


if __name__ == "__main__":
    asyncio.run(main())
