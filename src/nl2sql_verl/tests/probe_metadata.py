"""Record reproducible probe arguments and source hashes without environment secrets."""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def record_probe(args):
    if int(os.environ.get('RANK','0')) != 0:
        return
    root=Path(__file__).resolve().parents[3]
    files=['src/nl2sql_verl/fsdp_attention.py','src/nl2sql_verl/fsdp_engine.py','src/nl2sql_verl/fsdp_projection.py',
           'src/nl2sql_verl/tests/fsdp_cp_probe.py','src/nl2sql_verl/tests/fsdp_memory_probe.py',
           'src/nl2sql_verl/worker_python.sh']
    keys=['CUDA_VISIBLE_DEVICES','WORLD_SIZE','NL2SQL_ATTENTION_FP32',
          'NL2SQL_ATTENTION_BF16_IO','NL2SQL_FLEX_FP32_PRECISION',
          'NL2SQL_FLEX_CANONICAL_LAYOUT','NL2SQL_FLEX_STATIC','TORCHINDUCTOR_COMPILE_THREADS','OMP_NUM_THREADS']
    result={'recorded_utc':datetime.now(timezone.utc).isoformat(),'argv':sys.argv,
            'arguments':vars(args),'environment':{k:os.environ[k] for k in keys if k in os.environ},
            'source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files}}
    path=Path(str(args.output)+'.launch.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
