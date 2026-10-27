"""Compare attention on identical operands captured from pretrained HF Gemma."""
import os,json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
from transformers import Gemma4ForConditionalGeneration
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from src.nl2sql_verl.tests.fsdp_cp_probe import MODEL,fixture
from src.nl2sql_verl.fsdp_attention import attention,make_masks


def main():
    torch.set_num_threads(4);torch.cuda.set_device(0);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.preferred_blas_library('cublaslt');torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=(False,False)
    model=Gemma4ForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval()
    original=ALL_ATTENTION_FUNCTIONS['sdpa'];captured={}
    def capture(module,q,k,v,mask,**kw):
        out=original(module,q,k,v,mask,**kw)
        if getattr(module,'layer_idx',-1) in (0,5,59):captured[module.layer_idx]=(module,q.detach(),k.detach(),v.detach(),mask,out[0].detach(),kw)
        return out
    ALL_ATTENTION_FUNCTIONS.register('sdpa',capture)
    ids=fixture(MODEL,[257])[0][0].cuda()[None];pos=torch.arange(ids.shape[1],device='cuda')[None]
    with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):model(input_ids=ids,position_ids=pos,use_cache=False)
    masks=make_masks(pos,torch.ones_like(pos,dtype=torch.bool),1024)
    results=[]
    for layer,(module,q,k,v,mask,ref,kw) in captured.items():
        print('CAPTURE',layer,q.dtype,k.dtype,v.dtype,q.shape,k.shape,kw,'mask',None if mask is None else (mask.shape,mask.dtype),flush=True)
        block=masks['sliding_attention' if module.is_sliding else 'full_attention']
        for fp32 in ('0','1'):
            os.environ['NL2SQL_ATTENTION_FP32']=fp32
            with torch.no_grad():out,_=attention(module,q,k,v,block,**kw)
            row={'layer':layer,'fp32_attention':fp32=='1','relative_l2':float((out.float()-ref.float()).norm()/ref.float().norm()),'max_abs':float((out.float()-ref.float()).abs().max()),'q_std':float(q.float().std()),'k_std':float(k.float().std()),'v_std':float(v.float().std())}
            results.append(row);print(json.dumps(row),flush=True)
        torch.save({'q':q.cpu(),'k':k.cpu(),'v':v.cpu(),'mask':None if mask is None else mask.cpu(),'output':ref.cpu(),'scaling':kw['scaling']},f'outputs/fsdp_cp_recipe/attention_capture_layer{layer}.pt')
    Path('outputs/fsdp_cp_recipe/attention_capture.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
