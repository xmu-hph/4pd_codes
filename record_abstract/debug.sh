docker run --rm -it --gpus '"device=6"' --network host \
-v /mnt/data/liyihao/model_zoo/model-qwen2-7b-instruct:/root/model/big \
-v /mnt/data/hupenghui/model/zh_core_web_sm-3.7.0/zh_core_web_sm/zh_core_web_sm-3.7.0:/root/model/zh \
harbor-contest.4pd.io/hupenghui/summarize:qwen-spacy /bin/bash

#uvicorn.run("app:app", host="0.0.0.0", port=80)