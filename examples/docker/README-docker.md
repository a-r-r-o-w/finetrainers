# finetrainers Docker Setup Guide (for Windows)

This guide explains how to run `finetrainers` using Docker on a Windows environment.

## Prerequisites

- **Windows 10/11**
- **[Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)** is installed
- **NVIDIA GPU** and the latest NVIDIA drivers
- **WSL2** is enabled (can be set during Docker Desktop installation)

## Important: Docker Desktop Settings

Make sure Docker Desktop is configured correctly:

1. Confirm Docker Desktop is running (the icon appears in the task tray)
2. If not running, start "Docker Desktop" from the Start menu
3. Open Docker Desktop settings:
   - Right-click the Docker icon in the task tray
   - Select "Settings"
4. Check the following settings:
   - Under "General", ensure "Use WSL 2 based engine" is checked
   - Under "Resources" → "WSL Integration", ensure WSL2 is enabled
   - Under "Docker Engine", make sure the following is included in the JSON settings:
```json
{
  "experimental": true,
  "features": {
    "buildkit": true
  }
}
```

## Installation and Setup

### 1. Install Docker Desktop for Windows

1. Download the installer from [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. Run the installer and follow the instructions
3. During installation, select the option to "Use WSL2 Backend"
4. After installation, launch Docker Desktop
5. If you have trouble starting Docker Desktop:
   - Restart Windows
   - Make sure WSL2 is installed correctly (`wsl --status`)
   - Ensure Hyper-V is enabled

### 2. Set Up NVIDIA GPU Support

Special configuration is required to run NVIDIA containers on Windows:

1. Open Docker Desktop settings
2. Go to "Resources" → "WSL Integration"
3. Ensure your WSL2 distribution (e.g., Ubuntu) is enabled
4. Under "Settings" → "General", ensure "Use the WSL 2 based engine" is enabled
5. Open a WSL2 terminal and run the following commands:

```bash
# Run inside WSL2 terminal
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Not needed if docker service is not running inside WSL2
# sudo systemctl restart docker
```

### 3. Clone the finetrainers Repository

```bash
# Can be run in WSL2 terminal or Windows PowerShell
git clone https://github.com/a-r-r-o-w/finetrainers.git
cd finetrainers
```

## Usage

### Build and Start the Docker Environment

```bash
# Run in the root directory of the repository
docker-compose build
docker-compose up -d
```

### Connect to the Container

```bash
docker exec -it finetrainers bash
```

This will launch an interactive bash shell, allowing you to execute commands inside the container.

### Example: Running Training

You can run the finetrainers command inside the container:

```bash
# Run inside the container
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

## Folder Structure

- `datasets/` - Directory to store datasets (mounted as a volume)
- `outputs/` - Directory to store training results (mounted as a volume)

These directories are shared with the host machine, so you can access them directly from Windows.

## Common Issues and Solutions

### 1. GPU Not Detected

Check if the GPU is recognized with the following command:

```bash
docker exec -it finetrainers nvidia-smi
```

If there are issues:
- Restart Docker Desktop
- Update NVIDIA Driver
- Restart WSL2

### 2. Out of Memory Errors

You can resolve this by increasing the WSL2 memory limit:

1. Create a `.wslconfig` file in your user folder (C:\Users\<username>\.wslconfig)
2. Add the following content:
```
[wsl2]
memory=16GB
swap=32GB
processors=8
```
3. Restart WSL2:
```
wsl --shutdown
```

### 3. Disk Space Issues

If the WSL2 VHD size becomes too large, optimization is needed:

1. Exit WSL2:
```
wsl --shutdown
```
2. Optimize the WSL2 disk image:
```
Optimize-VHD -Path <path>\ext4.vhdx -Mode Full
```

## Limitations

- Performance may be slightly lower on Windows compared to native Linux
- Some hardware-dependent features (e.g., special GPU driver optimizations) may not be available inside Docker containers
- Be aware of WSL2 memory limits (by default, 80% of host memory)

## Troubleshooting

If issues persist, please create an issue including the following information:
- Windows version
- Docker Desktop version
- GPU model and driver version
- Error messages and log details