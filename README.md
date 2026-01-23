# 安装 SAM2 ：
```bash
git clone https://github.com/facebookresearch/sam2.git sam2lib ## 或者可以使用 git submodule update --init --recursive 来获取sam2子模块

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