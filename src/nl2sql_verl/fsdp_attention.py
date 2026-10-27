"""Exact Gemma attention using Flex and verl's autograd Ulysses collectives.

Parameters stay in upstream HF Gemma; only attention is exchanged between
sequence and head partitions. No TP projections or ring-softmax backward.
"""
import os
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers import AttentionInterface
from transformers.integrations.flex_attention import flex_attention_forward
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from types import SimpleNamespace
from verl.utils.ulysses import (gather_seq_scatter_heads, gather_heads_scatter_seq,
                                get_ulysses_sequence_parallel_world_size)

BACKEND = 'nl2sql_flex_ulysses'
_FLEX = torch.compile(flex_attention, dynamic=True)
_MASK = torch.compile(create_block_mask, dynamic=False)
OPTIONS = {'BLOCK_M':32,'BLOCK_N':32,'BLOCK_M1':16,'BLOCK_N1':16,
           'BLOCK_M2':16,'BLOCK_N2':16,'num_stages':1,'num_warps':4,
           'FORCE_USE_FLEX_ATTENTION':True}


def make_masks(position_ids, valid, sliding_window):
    """Build document-aware masks without a materialized token-by-token matrix."""
    docs = (position_ids == 0).cumsum(-1)
    width = position_ids.shape[-1]
    def causal(b,h,q,k):
        # Flex can query indices beyond the real tensor at the final block.
        qi=q.clamp(max=width-1);ki=k.clamp(max=width-1)
        return ((q<width)&(k<width)&valid[b,qi]&valid[b,ki]
                &(docs[b,qi]==docs[b,ki])&(q>=k))
    def sliding(b,h,q,k):
        return causal(b,h,q,k)&(q-k<sliding_window)
    return {kind:_MASK(fn,B=position_ids.shape[0],H=None,Q_LEN=width,KV_LEN=width,
                      device=str(position_ids.device),BLOCK_SIZE=128)
            for kind,fn in [('full_attention',causal),('sliding_attention',sliding)]}


def attention(module,query,key,value,attention_mask,dropout=0.,scaling=None,**kwargs):
    if dropout:
        raise ValueError('Gemma recipe requires zero attention dropout')
    if attention_mask is None:
        raise ValueError('Explicit global document masks are required')
    # FP32 residuals make rotary embeddings FP32 under BF16 autocast. Flex
    # requires homogeneous Q/K/V dtypes; use the projection's compute dtype.
    original_dtype=query.dtype
    fp32_output=os.environ.get('NL2SQL_ATTENTION_BF16_IO')=='1'
    if fp32_output:
        query,key,value=(x.to(torch.bfloat16) for x in (query,key,value))
    elif os.environ.get('NL2SQL_ATTENTION_FP32')!='1':
        query,key=query.to(value.dtype),key.to(value.dtype)
    sp=get_ulysses_sequence_parallel_world_size()
    if query.shape[1]%sp or key.shape[1]%sp:
        raise ValueError('Ulysses size must divide the actual layer Q and KV head counts')
    query=gather_seq_scatter_heads(query,seq_dim=2,head_dim=1)
    key=gather_seq_scatter_heads(key,seq_dim=2,head_dim=1)
    value=gather_seq_scatter_heads(value,seq_dim=2,head_dim=1)
    dtype=original_dtype if fp32_output else query.dtype
    options=dict(OPTIONS)
    if os.environ.get('NL2SQL_ATTENTION_FP32')=='1':
        query,key,value=(x.float() for x in (query,key,value))
        options.update(BLOCK_M=16,BLOCK_N=16,FLOAT32_PRECISION="'tf32x3'")
    if module is None:
        module=SimpleNamespace(training=torch.is_grad_enabled())
    with torch.autocast('cuda',enabled=False):
        output,_=flex_attention_forward(module,query,key,value,attention_mask,
            scaling=scaling,kernel_options=options)
    output=output.transpose(1,2).to(dtype)
    output=gather_heads_scatter_seq(output,head_dim=1,seq_dim=2)
    return output.transpose(1,2).contiguous(),None


def make_hf_masks(position_ids, sliding_window):
    """Use upstream HF's exact mask construction for packed, right-padded text.

    All real tokens precede suffix padding. Causality prevents real queries
    seeing that suffix; resetting position IDs isolates packed documents.
    Passing a 2D mask here would disable HF's packed-document detection.
    """
    config=SimpleNamespace(_attn_implementation='flex_attention',sliding_window=sliding_window)
    dummy=torch.empty(*position_ids.shape,1,device=position_ids.device,dtype=torch.bfloat16)
    args=dict(config=config,inputs_embeds=dummy,attention_mask=None,
              past_key_values=None,position_ids=position_ids)
    return {'full_attention':create_causal_mask(**args),
            'sliding_attention':create_sliding_window_causal_mask(**args)}


AttentionInterface.register(BACKEND,attention)
