"""Tool declarations and system instructions, separate from dataset and wire formats.

The Gemma profile reproduces the supplied JSONL references. Other models must
supply their own system instructions; their tokenizer and rollout parser own
serialization/parsing of calls. A custom profile is not runtime certification.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from .consensus.prompts import SYSTEM_PROMPT_TEMPLATES
from .consensus.tool_calling import get_tool_definitions, tool_catalog_compact


@dataclass(frozen=True)
class ToolProfile:
    name: str
    system_prompt: str
    parser_format: str | None = None
    agent_name: str = 'nl2sql_tool_agent'

    def declarations(self):
        """Return semantic function schemas, without model-specific call tokens."""
        return deepcopy(get_tool_definitions(include_consensus=False))

    def messages(self, user_prompt):
        return [{'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': user_prompt}]

    def prompt_length(self, tokenizer, messages):
        # The tokenizer's chat template owns model-specific serialization.
        ids = tokenizer.apply_chat_template(
            messages, tools=self.declarations(), tokenize=True,
            add_generation_prompt=True, return_dict=False)
        return len(ids)


def load_tool_profile(name='gemma4', system_prompt_file=None):
    if name == 'gemma4_native_v1':
        if system_prompt_file is not None:
            raise ValueError('Use the custom profile to replace system instructions')
        from .gemma_native_profile import system_prompt
        return ToolProfile(name, system_prompt(), 'nl2sql_gemma4', 'nl2sql_native_tool_agent')
    elif name == 'gemma4':
        if system_prompt_file is not None:
            raise ValueError('Use the custom profile to replace system instructions')
        prompt = SYSTEM_PROMPT_TEMPLATES.replace(
            '{TOOL_CATALOG_COMPACT}', tool_catalog_compact()).strip()
    elif name == 'custom':
        if system_prompt_file is None:
            raise ValueError('custom requires a model-specific system prompt file')
        prompt = Path(system_prompt_file).read_text().strip()
        if not prompt:
            raise ValueError('System prompt must not be empty')
    else:
        raise ValueError(f'Unknown tool profile: {name}')
    return ToolProfile(name, prompt)
