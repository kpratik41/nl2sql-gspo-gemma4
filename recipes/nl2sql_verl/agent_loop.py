"""Use verl's native Gemma tool loop, exposing only generated text to rewards."""
from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import Gemma4ToolParser, ToolParser

from .gemma_arguments import parse_arguments
from .task import assistant_tokens


@ToolParser.register('nl2sql_gemma4')
class NL2SQLGemmaParser(Gemma4ToolParser):
    def _parse_arguments(self, args_str):
        return parse_arguments(args_str)


@register('nl2sql_tool_agent')
class NL2SQLToolAgentLoop(ToolAgentLoop):
    async def run(self, sampling_params, **kwargs):
        output = await super().run(sampling_params, **kwargs)
        text, count = assistant_tokens(output.response_ids, output.response_mask, self.tokenizer)
        output.extra_fields.update(assistant_text=text, generated_token_count=count)
        return output
