import asyncio
import sys
sys.path.insert(0, 'backend')

from database.models import get_db
from database.crud import get_all_mcp_servers

async def main():
    async with get_db() as db:
        servers = await get_all_mcp_servers(db)
        for s in servers:
            print(s)

asyncio.run(main())
