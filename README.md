# 安装 SAM2 ：
```bash
git submodule update --init --recursive

cd sam2lib
pip install -e .

cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

## 配置 conda 环境 (采用 environment.yaml 与 requirements.txt)
```bash
conda env create -f environment.yaml -n myconda
conda activate myconda
python -m pip install -r requirements.txt
```