# 安定したベースイメージを使用
FROM python:3.10-slim

# 必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# CUDA関連のパッケージとPyTorchのインストール
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 作業ディレクトリを設定
WORKDIR /app

# リポジトリ内のファイルをコンテナにコピー
COPY . /app

# 必要なPythonパッケージをインストール
RUN pip install --no-cache-dir -r requirements.txt

# 最新のdiffusersをソースからインストール（READMEの推奨に従い）
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers

# Python環境変数を設定
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# マルチプロセッシングの方法を設定
ENV PYTHONWARNINGS="ignore"

# 環境変数として注意プロバイダーを設定
ENV FINETRAINERS_ATTN_PROVIDER=native
ENV FINETRAINERS_ATTN_CHECKS=0

# エントリーポイントを設定（シェルを使用してコマンドを実行可能に）
ENTRYPOINT ["/bin/bash"]