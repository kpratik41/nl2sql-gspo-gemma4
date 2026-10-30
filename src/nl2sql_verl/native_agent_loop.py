"""Versioned native Gemma agent with actual execution and completion telemetry."""
from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentState

from .agent_loop import NL2SQLToolAgentLoop
from .consensus.sql_utils import extract_final_answer_sql
from .prepare_reference_data import sql_tokens

METRICS = ('tool_execution_count', 'tool_error_count', 'tool_success_count',
           'sqlite_query_success_count', 'final_sql_verified_exact',
           'assistant_generation_count', 'response_budget_reached',
           'native_call_start_count', 'native_call_end_count')


@register('nl2sql_native_tool_agent')
class NL2SQLNativeAgentLoop(NL2SQLToolAgentLoop):
    async def _handle_generating_state(self, agent_data, sampling_params, **kwargs):
        remaining = self.response_length-len(agent_data.response_mask)
        if remaining <= 0:
            agent_data.extra_fields['native_context_budget_reached'] = 1
            return AgentState.TERMINATED
        # Allocate the remaining cumulative budget, rather than a fresh 4K
        # generation on every turn that would subsequently be sliced away.
        sampling_params = {**sampling_params, 'max_tokens': min(remaining, sampling_params.get('max_tokens', remaining))}
        state = await super()._handle_generating_state(agent_data, sampling_params, **kwargs)
        turn = {'generated_length': len(agent_data.response_ids),
                'last_token_id': agent_data.response_ids[-1] if agent_data.response_ids else None,
                'native_starts': agent_data.response_ids.count(self.tokenizer.convert_tokens_to_ids('<|tool_call>')),
                'native_ends': agent_data.response_ids.count(self.tokenizer.convert_tokens_to_ids('<tool_call|>'))}
        agent_data.extra_fields.setdefault('native_generation_trace', []).append(turn)
        return state

    async def _handle_processing_tools_state(self, agent_data):
        state = await super()._handle_processing_tools_state(agent_data)
        if state == AgentState.TERMINATED:
            agent_data.extra_fields['native_context_budget_reached'] = 1
        return state

    async def run(self, sampling_params, **kwargs):
        profile = (kwargs.get('extra_info') or {}).get('tool_profile')
        if profile != 'gemma4_native_v1' or self.tool_parser_name != 'nl2sql_gemma4':
            raise ValueError('Native Gemma agent requires its versioned data and matching parser')
        output = await super().run(sampling_params, **kwargs)
        fields = output.extra_fields
        events = fields.get('native_tool_events', [])
        trace = fields.get('native_generation_trace', [])
        successful_queries = [e for e in events if e['name']=='sqlite_query' and e['success']]
        final = extract_final_answer_sql(fields['assistant_text'])
        fields.update(
            tool_execution_count=len(events), tool_error_count=sum(not e['success'] for e in events),
            tool_success_count=sum(e['success'] for e in events),
            sqlite_query_success_count=len(successful_queries),
            final_sql_verified_exact=int(bool(final) and any(
                sql_tokens(final)==sql_tokens(e['parameters'].get('sql', '')) for e in successful_queries)),
            assistant_generation_count=len(trace), response_budget_reached=int(
                len(output.response_ids)>=self.response_length or fields.get('native_context_budget_reached', False)),
            native_call_start_count=sum(t['native_starts'] for t in trace),
            native_call_end_count=sum(t['native_ends'] for t in trace))
        return output
