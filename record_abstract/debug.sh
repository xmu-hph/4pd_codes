#docker run --rm -it --gpus '"device=4,5,6,7"' --network host \
docker run --rm -it --gpus '"device=4,5,6,7"' \
harbor.4pd.io/lab-platform/pk_platform/model_services/basemodel_ability_hupenghui:english /bin/bash

#-v /mnt/data/hupenghui/model/zh_core_web_sm-3.7.0/zh_core_web_sm/zh_core_web_sm-3.7.0:/root/model/zh \
#-v /mnt/data/liyihao/model_zoo/model-qwen2-7b-instruct:/root/model/big \
#uvicorn.run("app:app", host="0.0.0.0", port=80)