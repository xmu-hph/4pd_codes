docker run --rm -it --gpus '"device=6"' --network host \
-e CUDA_DEVICE_MEMORY_LIMIT=8096m \
-e LD_PRELOAD=/usr/local/vgpu/libvgpu.so \
-v /home/hupenghui/vgpu/libvgpu.so:/usr/local/vgpu/libvgpu.so \
-v /home/hupenghui/vgpulock:/tmp/vgpulock \
<<<<<<< HEAD
<<<<<<< HEAD
harbor-contest.4pd.io/hupenghui/tts:tts_stream_clone_1655 /bin/bash
#-v /mnt/data/hupenghui/model/tts_models--multilingual--multi-dataset--xtts_v2:/root/model/tts \
#-v /mnt/data/hupenghui/model/zh_core_web_sm-3.7.0/zh_core_web_sm/zh_core_web_sm-3.7.0:/root/model/zh \
#-v /mnt/data/hupenghui/model/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm-3.7.1:/root/model/en \
=======
-v /mnt/data/hupenghui/model/tts_models--multilingual--multi-dataset--xtts_v2:/root/model/tts \
-v /mnt/data/hupenghui/model/zh_core_web_sm-3.7.0/zh_core_web_sm/zh_core_web_sm-3.7.0:/root/model/zh \
-v /mnt/data/hupenghui/model/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm-3.7.1:/root/model/en \
harbor-contest.4pd.io/hupenghui/tts:tts_stream_clone_1655 /bin/bash
>>>>>>> parent of 16fe679 (remove first null response)
=======
-v /mnt/data/hupenghui/model/tts_models--multilingual--multi-dataset--xtts_v2:/root/model/tts \
-v /mnt/data/hupenghui/model/zh_core_web_sm-3.7.0/zh_core_web_sm/zh_core_web_sm-3.7.0:/root/model/zh \
-v /mnt/data/hupenghui/model/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm-3.7.1:/root/model/en \
harbor-contest.4pd.io/hupenghui/tts:tts_stream_clone_without_zero_res /bin/bash
>>>>>>> parent of f57c1a6 (new)

#uvicorn.run("app:app", host="0.0.0.0", port=80)