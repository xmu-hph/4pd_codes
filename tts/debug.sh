docker run --rm -it --gpus '"device=6"' --network host \
-v /mnt/data/hupenghui/model/tts_models--multilingual--multi-dataset--xtts_v2:/root/model/tts \
-v /mnt/data/hupenghui/model/zh_core_web_sm-3.7.0/zh_core_web_sm/zh_core_web_sm-3.7.0:/root/model/zh \
-v /mnt/data/hupenghui/model/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm-3.7.1:/root/model/en \
harbor-contest.4pd.io/hupenghui/tts:tts_stream /bin/bash

#uvicorn.run("app:app", host="0.0.0.0", port=80)