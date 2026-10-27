"""Use verl's Gemma tool loop, exposing only generated text to rewards."""
import asyncio
import json
import logging
import re

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall, Gemma4ToolParser, ToolParser

from .gemma_arguments import parse_arguments
from .task import assistant_tokens


@ToolParser.register('nl2sql_gemma4')
class NL2SQLGemmaParser(Gemma4ToolParser):
    # The consensus prompt explicitly requests a bare call: line. Gemma emits
    # that form with the native closing token but can omit the opening token.
    # Require a line boundary or the native opener, and a complete closer.
    call_pattern = re.compile(r'(?:^|\n|<\|tool_call>)[ \t]*call:(\w+)\{(.*?)\}<tool_call\|>', re.S)

    def _parse_arguments(self, args_str):
        return parse_arguments(args_str)

    async def extract_tool_calls(self, responses_ids, tools=None):
        text = await asyncio.to_thread(self.tokenizer.decode, responses_ids, skip_special_tokens=False)
        if self.tokenizer.pad_token:
            text = text.replace(self.tokenizer.pad_token, '')
        calls, first = [], None
        for match in self.call_pattern.finditer(text):
            try:
                arguments = self._parse_arguments(match.group(2))
                calls.append(FunctionCall(name=match.group(1), arguments=json.dumps(arguments, ensure_ascii=False)))
                if first is None:
                    first = match.start()
            except ValueError as exc:
                logging.getLogger(__name__).warning('Invalid Gemma tool arguments: %s', exc)
        return (text[:first] if first is not None else text), calls


@register('nl2sql_tool_agent')
class NL2SQLToolAgentLoop(ToolAgentLoop):
    async def run(self, sampling_params, **kwargs):
        output = await super().run(sampling_params, **kwargs)
        text, count = assistant_tokens(output.response_ids, output.response_mask, self.tokenizer)
        output.extra_fields.update(assistant_text=text, generated_token_count=count)
        return output
