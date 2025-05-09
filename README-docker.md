# finetrainers Docker セットアップガイド (Windows用)

このガイドでは、Windows環境でDockerを使用して`finetrainers`を実行する方法を説明します。

## 前提条件

- **Windows 10/11**
- **[Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)** がインストールされていること
- **NVIDIA GPU** と最新のNVIDIAドライバー
- **WSL2** が有効になっていること (Docker Desktopのインストール時に設定可能)

## 重要：Docker Desktop の設定

Docker Desktop が正しく設定されていることを確認してください：

1. Docker Desktop が起動していることを確認（タスクトレイにアイコンが表示されている）
2. もし起動していない場合は、スタートメニューから「Docker Desktop」を起動
3. Docker Desktop の設定を開く：
   - タスクトレイのDockerアイコンを右クリック
   - 「Settings」を選択
4. 以下の設定を確認：
   - 「General」で「Use WSL 2 based engine」にチェックが入っていること
   - 「Resources」→「WSL Integration」でWSL2が有効になっていること
   - 「Docker Engine」でJSON設定に以下が含まれていること：
```json
{
  "experimental": true,
  "features": {
    "buildkit": true
  }
}
```

## インストールと設定

### 1. Docker Desktop for Windows のインストール

1. [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/) からインストーラーをダウンロード
2. インストーラーを実行し、指示に従ってインストール
3. インストール中に「WSL2 Backendを使用する」オプションを選択
4. インストール後、Docker Desktopを起動
5. Docker Desktopの起動に問題がある場合：
   - Windowsを再起動
   - WSL2が正しくインストールされていることを確認（`wsl --status`）
   - Hyper-Vが有効になっていることを確認

### 2. NVIDIA GPUサポートの設定

Windows上でNVIDIAコンテナを実行するには、特別な設定が必要です：

1. Docker Desktopの設定を開く
2. 「Resources」→「WSL Integration」を選択
3. 使用しているWSL2ディストリビューション（Ubuntu等）が有効になっていることを確認
4. Docker Desktopの「Settings」→「General」で「Use the WSL 2 based engine」が有効になっていることを確認
5. WSL2ターミナルを開き以下のコマンドを実行：

```bash
# WSL2ターミナル内で実行
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2

# WSL2内でdockerサービスが起動されていない場合は不要
# sudo systemctl restart docker
```

### 3. finetrainers リポジトリのクローン

```bash
# WSL2ターミナルまたはWindows PowerShellで実行可能
git clone https://github.com/a-r-r-o-w/finetrainers.git
cd finetrainers
```

## 使用方法

### Docker環境のビルドと起動

```bash
# リポジトリのルートディレクトリで実行
docker-compose build
docker-compose up -d
```

### コンテナへの接続

```bash
docker exec -it finetrainers bash
```

これにより、対話型のbashシェルが起動し、コンテナ内でコマンドを実行できます。

### 学習の実行例

コンテナ内でfinetrainersコマンドを実行できます：

```bash
# コンテナ内で実行
python train.py \
    --training_type lora \
    --model_name ltx_video \
    --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt \
    --dataset_type image_video \
    --train_data '{"path": "path/to/your/dataset/train", "video_column": "video", "caption_column": "text"}' \
    --resolution 576 \
    --timestep_range 0 1000 \
    --num_train_epochs 100 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --checkpointing_steps 100 \
    --learning_rate 5e-5 \
    --lr_scheduler constant \
    --lr_warmup_steps 0 \
    --output_dir outputs/ltx_video_lora \
    --validation_steps 50 \
    --dataloader_num_workers 16 \
    --mixed_precision bf16 \
    --enable_xformers_memory_efficient_attention \
    --seed 42
```

## フォルダ構造

- `datasets/` - データセットを保存するディレクトリ（ボリュームマウント済み）
- `outputs/` - 学習結果を保存するディレクトリ（ボリュームマウント済み）

これらのディレクトリは、ホストマシンと共有されるため、Windows側からも直接アクセス可能です。

## よくある問題と解決策

### 1. GPUが認識されない場合

以下のコマンドでGPUが認識されているか確認：

```bash
docker exec -it finetrainers nvidia-smi
```

問題がある場合：
- Docker Desktopの再起動
- NVIDIA Driverの更新
- WSL2の再起動

### 2. メモリ不足エラー

WSL2のメモリ制限を増やすことで解決できます：

1. ユーザーフォルダに`.wslconfig`ファイルを作成（C:\Users\<ユーザー名>\.wslconfig）
2. 以下の内容を追加：

```
[wsl2]
memory=16GB
swap=32GB
processors=8
```

3. WSL2を再起動：
```
wsl --shutdown
```

### 3. ディスク容量の問題

WSL2のVHDサイズが大きくなりすぎた場合、最適化が必要：

1. WSL2を終了：
```
wsl --shutdown
```

2. WSL2ディスクイメージを最適化：
```
Optimize-VHD -Path <パス>\ext4.vhdx -Mode Full
```

## 制限事項

- Windows環境では、Linuxネイティブに比べてパフォーマンスがわずかに低下する場合があります
- 特定のハードウェア依存機能（例：特殊なGPUドライバ最適化）がDockerコンテナ内では利用できない場合があります
- WSL2のメモリ制限に注意してください（デフォルトではホストメモリの80%）

## トラブルシューティング

問題が解決しない場合は、以下の情報を含むイシューを作成してください：
- Windowsバージョン
- Docker Desktopバージョン
- GPUモデルとドライバーバージョン
- エラーメッセージやログの詳細