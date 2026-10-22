"""Render a prepared record using the checkpoint's native tool chat template."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='data/prepared/california_schools.jsonl')
    p.add_argument('--model', default='/home/ec2-user/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/snapshots/a9904d24bcc1d289a1950fa9d2b978c47cf903b9')
    p.add_argument('--output', default='data/prepared/california_schools.prompt.txt')
    args = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    lengths = []
    for line in Path(args.input).read_text().splitlines():
        record = json.loads(line)
        prompt = tokenizer.apply_chat_template(record['messages'], tools=record['tools'], tokenize=False,
                    add_generation_prompt=True, enable_thinking=True, force_nonempty_content=True,
                    truncate_history_thinking=False)
        for name in ('sqlite_query', 'sqlite_peek', 'bm_25_search'):
            assert f'<name>{name}</name>' in prompt, name
        lengths.append(len(tokenizer.encode(prompt, add_special_tokens=False)))
        if len(lengths) == 1:
            Path(args.output).write_text(prompt)
    print(json.dumps({'records': len(lengths), 'min_prompt_tokens': min(lengths),
                      'max_prompt_tokens': max(lengths), 'first_prompt': args.output}))


if __name__ == '__main__':
    main()
