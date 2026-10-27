"""Right-padded, independent sequences for Gemma's model-owned attention masks."""
import math

import torch
from torch.nn.functional import pad


def pad_sequences(input_ids, position_ids, pad_id, cp_size=1, bucket_size=1):
    ids = input_ids.unbind()
    positions = position_ids.unbind()
    lengths = [row.numel() for row in ids]
    if not lengths or min(lengths) == 0:
        raise ValueError('Microbatches must contain nonempty sequences')
    if len(positions) != len(ids) or any(p.numel() != n for p, n in zip(positions, lengths)):
        raise ValueError('Position IDs must match each input sequence')
    unit = math.lcm(2 * cp_size, bucket_size)
    width = math.ceil(max(lengths) / unit) * unit
    tokens = torch.stack([pad(row, (0, width-n), value=pad_id) for row, n in zip(ids, lengths)])
    pos = torch.stack([pad(row, (0, width-n)) for row, n in zip(positions, lengths)])
    # Shift inside each sequence. Never predict the first token of another row.
    labels = torch.stack([pad(row.roll(-1), (0, width-n), value=pad_id) for row, n in zip(ids, lengths)])
    padding = torch.arange(width, device=tokens.device)[None] >= torch.tensor(lengths, device=tokens.device)[:, None]
    return {'input_ids': tokens, 'position_ids': pos, 'labels': labels, 'padding_mask': padding}, lengths


def unpad_values(values, lengths, offsets):
    flat = torch.cat([row[:n] for row, n in zip(values.unbind(), lengths, strict=True)])
    return torch.nested.nested_tensor_from_jagged(flat, offsets)


def batch_temperature(value, batch_size):
    if not isinstance(value, torch.Tensor) or value.numel() == 1:
        return value
    if value.numel() != batch_size:
        raise ValueError('Expected one temperature per trajectory')
    return value.reshape(batch_size, 1)
