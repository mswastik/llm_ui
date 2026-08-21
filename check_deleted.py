import sys, asyncio
from sqlalchemy import text
sys.path.insert(0, "backend")
from database.models import get_db

async def main():
    async with get_db() as db:
        for cid in ['3939c97c', '79215204']:
            q = text("SELECT id, title FROM conversations WHERE id LIKE :c")
            row = await db.execute(q, {"c": cid + "%"})
            res = row.fetchall()
            print(cid, '->', ('FOUND: ' + str(res[0][1])) if res else 'DELETED')

asyncio.run(main())
