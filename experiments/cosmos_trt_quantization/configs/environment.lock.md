# 实验依赖锁定

- 主项目环境（只读复用）：`/home/rxhuang/Projects/cosmos-policy/.venv`
- 隔离实验环境：`/data/rxhuang/envs/cosmos-trt`
- Python：3.10
- PyTorch：2.7.0+cu128（来自主项目环境）
- TensorRT：10.9.0.34
- Torch-TensorRT：2.7.0
- NVIDIA ModelOpt：0.27.0
- ONNX：1.17.0
- ONNX Runtime GPU：1.20.1

安装命令：

```bash
uv venv --python /home/rxhuang/Projects/cosmos-policy/.venv/bin/python \
  /data/rxhuang/envs/cosmos-trt
UV_CACHE_DIR=/data/rxhuang/uv_cache uv pip install \
  --python /data/rxhuang/envs/cosmos-trt/bin/python \
  'tensorrt-cu12==10.9.0.34'
UV_CACHE_DIR=/data/rxhuang/uv_cache uv pip install \
  --python /data/rxhuang/envs/cosmos-trt/bin/python --no-deps \
  'torch-tensorrt==2.7.0' 'nvidia-modelopt==0.27.0'
UV_CACHE_DIR=/data/rxhuang/uv_cache uv pip install \
  --python /data/rxhuang/envs/cosmos-trt/bin/python --no-deps \
  'onnx==1.17.0' 'onnxruntime-gpu==1.20.1' 'onnxscript==0.2.2' \
  'onnx-ir==0.1.9' 'pulp==2.9.0' 'torchprofile==0.0.4'
```

隔离环境通过 `.pth` 只读复用主环境依赖。没有升级或替换主环境中的
PyTorch、CUDA、torchao、TransformerEngine。
