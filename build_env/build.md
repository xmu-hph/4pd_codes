# 无模型基础镜像
```sh
harbor.4pd.io/lab-platform/pk_platform/model_services/hph_base:jupyter_nvcc_torch_ubuntu2204_base_image_vscode
```

# 启动镜像（映射内存、网络、cpu）
```sh
docker run --rm -it --gpus '"device=6"' --network host \
-v /mnt/data/hupenghui/model/tts_models--multilingual--multi-dataset--xtts_v2:/root/model/tts \
-v /mnt/data/hupenghui/model/zh_core_web_sm-3.7.0/zh_core_web_sm/zh_core_web_sm-3.7.0:/root/model/zh \
-v /mnt/data/hupenghui/model/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm-3.7.1:/root/model/en \
harbor-contest.4pd.io/hupenghui/tts:tts_stream /bin/bash
```

# 启动镜像（compose）
```yaml
version: '3.7'
services:
  tts:
    image: harbor.4pd.io/lab-platform/pk_platform/model_services/hph_for_4pd_tts:jupyter_nvcc_torch230_cuda121_cudnn8_ubuntu2204_xtts_model_image_base_vscode
    container_name: tts_hupenghui
    privileged: true
    network_mode: host
    pid: host
    ipc: host
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    volumes:
      - /home/hupenghui/code:/home/mnt
    devices:
      - /dev/fuse:/dev/fuse
    cap_add:
      - SYS_ADMIN
      - MKNOD
      - NET_ADMIN
    security_opt:
      - apparmor:unconfined
    stdin_open: true
    tty: true
    labels:
      com.example.category: tts_hupenghui
      com.example.maintainer: hupenghui
  traffic:
    image: harbor.4pd.io/lab-platform/pk_platform/model_services/hph_for_automl:traffic_nvcc_torch210_cuda118_cudnn8_ubuntu2204_vscode_base_image
    container_name: traffic_hupenghui
    privileged: true
    network_mode: host
    pid: host
    ipc: host
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    volumes:
      - /home/hupenghui/code:/home/mnt
    devices:
      - /dev/fuse:/dev/fuse
    cap_add:
      - SYS_ADMIN
      - MKNOD
      - NET_ADMIN
    security_opt:
      - apparmor:unconfined
    stdin_open: true
    tty: true
    labels:
      com.example.category: traffic_hupenghui
      com.example.maintainer: hupenghui
  download:
    image: harbor.4pd.io/lab-platform/pk_platform/model_services/hph_base:jupyter_nvcc_torch_ubuntu2204_base_image_vscode
    container_name: download_hupenghui
    privileged: true
    network_mode: host
    pid: host
    ipc: host
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    volumes:
      - /mnt/data:/home/mnt
    devices:
      - /dev/fuse:/dev/fuse
    cap_add:
      - SYS_ADMIN
      - MKNOD
      - NET_ADMIN
    security_opt:
      - apparmor:unconfined
    stdin_open: true
    tty: true
    labels:
      com.example.category: download_hupenghui
      com.example.maintainer: hupenghui
```
注：问`chatgpt`上述命令中分别映射了什么内容。是内存、网络还是cpu、gpu等。

# 使用代理
1. 同一个公网`ip`局域网下的机器，如果一个机器在`7890`端口开了代理，那么另一台机器可以使用`export https_proxy=http://10.100.116.56:7890 http_proxy=http://10.100.116.56:7890 all_proxy=socks5://10.100.116.56:7890`这种方式使用代理，不用进行端口映射，只是端口转发而已。
2. 具有不同公网`ip`的机器，是不能通过上述转发的方式使用代理。在具有代理的机器上使用`ssh`端口映射将代理端口映射到无代理机器上，这样`export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890`也是可以用的。