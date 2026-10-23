"""Adapt the three active consensus SQLite tools to verl's BaseTool contract."""
import asyncio
import inspect
import json
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

from .consensus import gen_tools
from .task import database_path


class SQLiteTool(BaseTool):
    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        self._instances = {}
        self._slots = asyncio.Semaphore(int(config.get('concurrency', 4)))

    async def create(self, instance_id=None, **kwargs):
        db_id = kwargs.get('create_kwargs', {}).get('db_id')
        path = database_path(db_id)
        instance_id = instance_id or str(uuid4())
        self._instances[instance_id] = (db_id, path)
        return instance_id, ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        expected, path = self._instances[instance_id]
        if parameters.get('db_id') != expected:
            return ToolResponse(text='Tool db_id must match this example.'), 0., {'tool_error': 1.}
        fn = getattr(gen_tools, '_'+self.name+'_impl')
        allowed = inspect.signature(fn).parameters
        unknown = set(parameters)-set(allowed)
        if unknown:
            return ToolResponse(text=f'Unknown arguments: {sorted(unknown)}'), 0., {'tool_error': 1.}
        # A process-wide immutable DB resolver is installed below. Roots and query
        # budgets are server-owned; arguments may request stricter budgets only.
        args = dict(parameters)
        budget = float(self.config.get('timeout_s', 60))
        args['timeout_s'] = min(float(args.get('timeout_s') or budget), budget)
        args['vm_step_limit'] = min(int(args.get('vm_step_limit') or 15_000_000), 15_000_000)
        args['busy_timeout_ms'] = min(int(args.get('busy_timeout_ms') or 5000), 5000)
        if self.name == 'sqlite_query':
            args['max_return_rows'] = min(int(args.get('max_return_rows') or 100), 100)
            args.pop('limit', None)
        if self.name == 'sqlite_peek':
            args['limit'] = min(int(args.get('limit') or 10), 100)
        if self.name == 'bm25_search_sqlite':
            args['top_k'] = min(int(args.get('top_k') or 10), 100)
        async with self._slots:
            result = await asyncio.wait_for(asyncio.to_thread(fn, **args), timeout=budget+5)
        # Tool outcomes are observations, not an additional reward channel.
        return ToolResponse(text=json.dumps(result, ensure_ascii=False, default=str)), 0., {}

    async def release(self, instance_id, **kwargs):
        self._instances.pop(instance_id, None)


# Preserve branch tool behavior while restricting path resolution to configured roots.
gen_tools._resolve_db_path = lambda db_id: str(database_path(db_id))
