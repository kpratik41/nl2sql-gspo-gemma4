"""Diagnostic CP2 model comparison replacing only Flex with SDPA math."""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from src.nl2sql_verl import fsdp_attention


def sdpa_attention(module, query, key, value, attention_mask,
                   scaling=None, dropout=0., **kwargs):
    q = torch.arange(query.shape[2], device=query.device)[:,None]
    k = torch.arange(key.shape[2], device=key.device)[None,:]
    allowed = attention_mask.mask_mod(0,0,q,k)
    with sdpa_kernel(SDPBackend.MATH):
        out = F.scaled_dot_product_attention(query,key,value,attn_mask=allowed[None,None],
                                             scale=scaling,dropout_p=dropout,enable_gqa=True)
    return out.transpose(1,2),None


if __name__ == '__main__':
    fsdp_attention.flex_attention_forward = sdpa_attention
    from src.nl2sql_verl.tests.fsdp_cp_probe import main
    main()
