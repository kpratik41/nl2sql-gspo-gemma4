# Environment Snapshot For Replication

Captured on 2026-07-02 from `/home/ubuntu/consensus`.

This machine has host CUDA 13.2 installed, but the Python training environment uses the PyPI CUDA 12.8 PyTorch/vLLM stack. For vLLM rollout reproducibility, replicate the Python wheel stack first; the host driver only needs to be new enough for CUDA 12.8/13.x and the same GPU/NVLink/NCCL topology if possible.

## Quick Rebuild Recipe

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.10.0 \
  transformers==5.12.1 \
  datasets==5.0.0 \
  accelerate==1.14.0 \
  deepspeed==0.19.2 \
  trl==1.4.0 \
  vllm==0.19.1 \
  wandb==0.28.0 \
  sentencepiece==0.2.1 \
  protobuf==6.33.6 \
  rank-bm25==0.2.2
```

Validation:

```bash
python - <<'PY'
import torch, vllm, flashinfer, deepspeed
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
print("vllm", vllm.__version__)
print("flashinfer", flashinfer.__version__)
print("deepspeed", deepspeed.__version__)
PY
pip check
vllm collect-env
```

## 1. Requested pip freeze subset

```text
accelerate==1.14.0
deepspeed==0.19.2
flashinfer-cubin==0.6.6
flashinfer-python==0.6.6
nvidia-cublas-cu12==12.8.4.1
nvidia-cudnn-cu12==9.10.2.21
nvidia-cudnn-frontend==1.18.0
nvidia-nccl-cu12==2.27.5
torch==2.10.0
torch_c_dlpack_ext==0.1.5
torchaudio==2.10.0
torchvision==0.25.0
transformers==5.12.1
triton==3.6.0
trl==1.4.0
vllm==0.19.1
```

`xformers` is not installed.

## 2. PyTorch build flavor and bundled CUDA/NCCL

```text
torch.__version__              = 2.10.0+cu128
torch.version.cuda             = 12.8
torch.version.git_version      = 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0
torch.cuda.is_available        = True
torch.cuda.device_count        = 8
torch.backends.cudnn.version   = 92000
torch.cuda.nccl.version        = (2, 27, 5)
```

Important bundled PyPI CUDA packages:

```text
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.5
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvshmem-cu12==3.4.5
nvidia-nvtx-cu12==12.8.90
triton==3.6.0
```

PyTorch build details:

```text
PyTorch built with GCC 13.3
C++ standard: 201703
CUDA Runtime: 12.8
CuDNN: 9.2.0 reported at runtime; build settings show CUDNN_VERSION=9.10.2
NCCL: enabled
CUDA arch flags include sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120
CPU capability usage: AVX512
BLAS: Intel oneAPI MKL 2024.2
MKL-DNN: v3.7.1
```

## 3. vLLM version and install method

```text
vllm==0.19.1
import version: 0.19.1
installed by: pip
wheel tag: cp38-abi3-linux_x86_64
source build: no direct_url.json present; this is a prebuilt PyPI wheel
package path: .venv/lib/python3.12/site-packages/vllm
```

`vllm collect-env` reports:

```text
vLLM Version: 0.19.1
vLLM Build Flags:
  CUDA Archs: Not Set
  ROCm: Disabled
CUDA used to build PyTorch: 12.8
CUDA runtime version visible to vLLM: 13.2.51
NVIDIA driver: 595.71.05
```

## 4. GPU driver, host CUDA, and topology

GPU:

```text
8 x NVIDIA H200
Memory per GPU from nvidia-smi: 143771 MiB
Memory per GPU from torch: 150111977472 bytes
Compute capability: 9.0
SM count per GPU: 132
MIG: disabled
Compute mode: Default
```

Driver and `nvidia-smi`:

```text
NVIDIA-SMI: 595.71.05
Driver Version: 595.71.05
Driver-reported CUDA Version: 13.2
```

Host CUDA toolkit:

```text
/usr/local/cuda -> /usr/local/cuda-13.2
nvcc: 13.2.51
CUDA SDK: 13.2.20260303
cudart: 13.2.51
nvrtc: 13.2.51
CUPTI: 13.2.23
cuBLAS: 13.3.0.5
cuFFT: 12.2.0.37
cuRAND: 10.4.2.51
cuSOLVER: 12.1.0.51
cuSPARSE: 12.7.9.17
GPUDirect Storage / cufile: 1.17.0.44
NPP: 13.1.0.44
nvJitLink: 13.2.51
system cuDNN headers/libs: 9.20.0
system NCCL headers/libs: 2.29.7
```

Topology:

```text
All GPU pairs are connected via NV18.
GPU0-3 CPU affinity: 0-47,96-143, NUMA node 0
GPU4-7 CPU affinity: 48-95,144-191, NUMA node 1
NVIDIA Fabric Manager: active
```

Relevant kernel modules:

```text
nvidia
nvidia_uvm
nvidia_modeset
nvidia_fs
gdrdrv
efa_nv_peermem
```

## 5. FlashInfer build details

```text
flashinfer-python==0.6.6
flashinfer-cubin==0.6.6
import name: flashinfer
flashinfer.__version__: 0.6.6
flashinfer git version: 70b142b75b46aa56e7f675a8e6ec1a977352c91f
flashinfer_cubin.__version__: 0.6.6
installed by: pip
wheel tags:
  flashinfer-python: py3-none-any
  flashinfer-cubin: py3-none-any
precompiled cubin files present: 9734
top-level cubin hash dirs:
  1fddc48b7b48af33914d040051b3e2ee9ba4701e
  a72d85b019dc125b9f711300cb989430f762f5a6
  b55211623be7f5697c5262ffd8361fc06c147bc9
  f1ed60e5666a7620683a8c34a41c850a25029b35
```

Observed vLLM FlashInfer-related defaults from `vllm.envs`:

```text
VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=True
VLLM_FLASHINFER_ALLREDUCE_BACKEND=auto
VLLM_FLASHINFER_MOE_BACKEND=latency
VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=413138944
VLLM_ALLREDUCE_USE_FLASHINFER=False
VLLM_USE_FLASHINFER_MOE_FP16=False
VLLM_USE_FLASHINFER_MOE_FP4=False
VLLM_USE_FLASHINFER_MOE_FP8=False
VLLM_USE_FLASHINFER_MOE_INT4=False
VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=False
VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=False
VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS=False
VLLM_USE_FLASHINFER_SAMPLER=None
VLLM_HAS_FLASHINFER_CUBIN=False
```

Note: `flashinfer_cubin` imports successfully even though `vllm.envs` reports `VLLM_HAS_FLASHINFER_CUBIN=False`. If the other cluster differs here, it may affect backend selection.

## 6. Environment overrides

Shell environment did not have explicit `NCCL_*`, `CUDA_VISIBLE_DEVICES`, or `VLLM_*` overrides set.

Relevant shell variables observed:

```text
LD_LIBRARY_PATH contains, repeated several times:
  /opt/amazon/openmpi/lib
  /usr/local/cuda/lib
  /usr/local/cuda
  /usr/local/cuda/lib64
  /usr/local/cuda/extras/CUPTI/lib64
  /usr/local/cuda/targets/x86_64-linux/lib
  /usr/local/lib
  /usr/lib

PATH contains:
  /usr/local/cuda/bin
  /usr/local/cuda/include
  /opt/amazon/openmpi/bin
  /opt/amazon/efa/bin
```

Relevant variables visible inside Python:

```text
PYTORCH_NVML_BASED_CUDA_CHECK=1
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_ubuntu
TORCHINDUCTOR_COMPILE_THREADS=1
TRITON_CACHE_AUTOTUNING=1
```

`vllm collect-env` additionally displayed:

```text
NO_COLOR=1
VLLM_WORKER_MULTIPROC_METHOD=spawn
```

But a direct shell `env` did not show `VLLM_WORKER_MULTIPROC_METHOD`; treat it as a vLLM collect-env/runtime observation, not a global shell export.

## Host OS and build tools

```text
OS: Ubuntu 24.04.4 LTS (Noble)
Kernel: Linux 6.17.0-1019-aws
glibc: 2.39 (Ubuntu GLIBC 2.39-0ubuntu8.7)
Python: 3.12.3
pip: 26.1.2
gcc: 13.3.0
g++: 13.3.0
cmake: 3.28.3
make: 4.3
ninja: 1.13.0 from the Python venv
```

CPU and memory:

```text
CPU: Intel Xeon Platinum 8488C
Sockets: 2
Cores per socket: 48
Threads per core: 2
Logical CPUs: 192
NUMA nodes: 2
Memory: 2.0 TiB
Swap: 0
```

AWS/distributed packages:

```text
efa: 3.0.0-1.amzn1
efa-config: 1.18
efa-nv-peermem: 1.2.3-1.amzn1
efa-profile: 1.7
libfabric AWS: 2.4.0amzn1.0
libnccl-ofi: 1.18.0-1
Open MPI active command reports: 4.1.7
openmpi40-aws package: 4.1.7-1
openmpi50-aws package: 5.0.9
DCGM: 4.6.0
NVIDIA container toolkit: 1.19.1
nvidia-fabricmanager: 595.71.05-1ubuntu1
```

## Full pip freeze

```text
accelerate==1.14.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.1
aiosignal==1.4.0
annotated-doc==0.0.4
annotated-types==0.7.0
anthropic==0.115.1
anyio==4.14.1
apache-tvm-ffi==0.1.12
astor==0.8.1
attrs==26.1.0
blake3==1.0.9
cachetools==7.1.4
cbor2==6.1.2
certifi==2026.6.17
cffi==2.0.0
charset-normalizer==3.4.7
click==8.4.2
cloudpickle==3.1.2
compressed-tensors==0.15.0.1
cryptography==49.0.0
cuda-bindings==12.9.4
cuda-pathfinder==1.5.6
cuda-python==12.9.4
datasets==5.0.0
deepspeed==0.19.2
depyf==0.20.0
detect-installer==0.1.0
dill==0.4.1
diskcache==5.6.3
distro==1.9.0
dnspython==2.8.0
docstring_parser==0.18.0
einops==0.8.2
email-validator==2.3.0
fastapi==0.139.0
fastapi-cli==0.0.27
fastapi-cloud-cli==0.22.1
fastar==0.11.0
filelock==3.29.4
flashinfer-cubin==0.6.6
flashinfer-python==0.6.6
frozenlist==1.8.0
fsspec==2026.4.0
gguf==0.19.0
gitdb==4.0.12
GitPython==3.1.50
googleapis-common-protos==1.75.0
grpcio==1.81.1
h11==0.16.0
hf-xet==1.5.1
hjson==3.1.0
httpcore==1.0.9
httptools==0.8.0
httpx==0.28.1
httpx-sse==0.4.3
huggingface_hub==1.21.0
idna==3.18
ijson==3.5.0
interegular==0.3.3
Jinja2==3.1.6
jiter==0.16.0
jmespath==1.1.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
lark==1.2.2
llguidance==1.3.0
llvmlite==0.44.0
lm-format-enforcer==0.11.3
loguru==0.7.3
markdown-it-py==4.2.0
MarkupSafe==3.0.3
mcp==1.28.1
mdurl==0.1.2
mistral_common==1.11.5
model-hosting-container-standards==0.1.16
mpmath==1.3.0
msgpack==1.2.1
msgspec==0.21.1
multidict==6.7.1
multiprocess==0.70.19
networkx==3.6.1
ninja==1.13.0
numba==0.61.2
numpy==2.2.6
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvdisasm==13.3.73
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cudnn-frontend==1.18.0
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-cutlass-dsl==4.6.0
nvidia-cutlass-dsl-libs-base==4.6.0
nvidia-cutlass-dsl-libs-core==4.6.0
nvidia-cutlass-dsl-libs-cu12==4.6.0
nvidia-ml-py==13.610.43
nvidia-nccl-cu12==2.27.5
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvshmem-cu12==3.4.5
nvidia-nvtx-cu12==12.8.90
openai==2.44.0
openai-harmony==0.0.8
opencv-python-headless==5.0.0.93
opentelemetry-api==1.43.0
opentelemetry-exporter-otlp==1.43.0
opentelemetry-exporter-otlp-proto-common==1.43.0
opentelemetry-exporter-otlp-proto-grpc==1.43.0
opentelemetry-exporter-otlp-proto-http==1.43.0
opentelemetry-proto==1.43.0
opentelemetry-sdk==1.43.0
opentelemetry-semantic-conventions==0.64b0
opentelemetry-semantic-conventions-ai==0.5.1
outlines_core==0.2.11
packaging==26.2
pandas==3.0.3
partial-json-parser==0.2.1.1.post7
pillow==12.3.0
platformdirs==4.10.0
prometheus-fastapi-instrumentator==8.0.2
prometheus_client==0.25.0
propcache==0.5.2
protobuf==6.33.6
psutil==7.2.2
py-cpuinfo==9.0.0
pyarrow==24.0.0
pybase64==1.4.3
pycountry==26.2.16
pycparser==3.0
pydantic==2.13.4
pydantic-extra-types==2.11.1
pydantic-settings==2.14.2
pydantic_core==2.46.4
Pygments==2.20.0
PyJWT==2.13.0
python-dateutil==2.9.0.post0
python-dotenv==1.2.2
python-json-logger==4.1.0
python-multipart==0.0.32
PyYAML==6.0.3
pyzmq==27.1.0
quack-kernels==0.5.0
rank-bm25==0.2.2
referencing==0.37.0
regex==2026.6.28
requests==2.34.2
rich==15.0.0
rich-toolkit==0.20.1
rignore==0.7.6
rpds-py==2026.6.3
safetensors==0.8.0
sentencepiece==0.2.1
sentry-sdk==2.64.0
setproctitle==1.3.7
setuptools==80.10.2
shellingham==1.5.4
six==1.17.0
smmap==5.0.3
sniffio==1.3.1
sse-starlette==3.4.5
starlette==1.3.1
supervisor==4.3.0
sympy==1.14.0
tabulate==0.10.0
tiktoken==0.13.0
tokenizers==0.22.2
torch==2.10.0
torch_c_dlpack_ext==0.1.5
torchaudio==2.10.0
torchvision==0.25.0
tqdm==4.68.3
transformers==5.12.1
triton==3.6.0
trl==1.4.0
typer==0.25.1
typing-inspection==0.4.2
typing_extensions==4.16.0
urllib3==2.7.0
uvicorn==0.49.0
uvloop==0.22.1
vllm==0.19.1
wandb==0.28.0
watchfiles==1.2.0
websockets==16.0
wheel==0.47.0
xgrammar==0.2.3
xxhash==3.8.0
yarl==1.24.2
```
