"""Observed tools for the versioned native profile; no change to legacy workers."""
import json

from .tools import SQLiteTool


class ObservedSQLiteTool(SQLiteTool):
    async def execute(self, instance_id, parameters, **kwargs):
        data = kwargs.get('agent_data')
        event = {'name': self.name, 'parameters': dict(parameters), 'success': False}
        if data is not None:
            data.extra_fields.setdefault('native_tool_events', []).append(event)
        try:
            response, reward, extra = await super().execute(instance_id, parameters, **kwargs)
            try:
                decoded = json.loads(response.text)
            except (ValueError, TypeError):
                decoded = None
            event['success'] = not extra.get('tool_error') and decoded is not None and not (
                isinstance(decoded, dict) and bool(decoded.get('error')))
            event['observation'] = (response.text or '')[:2048]
            return response, reward, extra
        except Exception as error:
            event['observation'] = f'{type(error).__name__}: {error}'
            raise
