# 基础镜像：NVIDIA CUDA Runtime + Ubuntu 22.04
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
# 避免 apt 交互提示
ENV DEBIAN_FRONTEND=noninteractive

# 安装基础工具（wget / bzip2 / git）
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda 到 /opt/conda
ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && $CONDA_DIR/bin/conda clean -afy

# 把 conda 加到 PATH 里
ENV PATH=$CONDA_DIR/bin:$PATH
SHELL ["bash", "-lc"]

# 设置工作目录，配置environment.yaml
WORKDIR /app
COPY environment.yaml /tmp/environment.yaml

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda env create -f /tmp/environment.yaml \
    && conda clean -afy \
    && rm /tmp/environment.yaml

# 默认启用 myconda 环境（不用每次手动 conda activate）
ENV CONDA_DEFAULT_ENV=myconda
ENV PATH=$CONDA_DIR/envs/$CONDA_DEFAULT_ENV/bin:$PATH
SHELL ["bash", "-lc"]

# 拷贝项目代码
COPY . /app

# CMD ["python", "test.py"]