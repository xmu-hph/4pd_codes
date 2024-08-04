FROM harbor.4pd.io/lab-platform/pk_platform/model_services/qwen2_7b_qinyu:base_20240607
#FROM harbor.4pd.io/lab-platform/pk_platform/model_services/hph_base:jupyter_nvcc_torch_ubuntu2204_base_image_vscode
#FROM nvcr.io/nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
ARG HTTP_PROXY=http://10.100.116.50:7890
ARG HTTPS_PROXY=http://10.100.116.50:7890
ARG all_proxy=socks5://10.100.116.50:7890
RUN pip install websockets wave aiohttp librosa
RUN pip install loguru flask_socketio eventlet fastapi pydantic uvicorn
RUN pip install spacy spacy-pkuseg transformers -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
RUN pip install accelerate
RUN mkdir /home/code
WORKDIR /home/code
RUN mkdir -p /root/model/zh
COPY zh_core_web_sm-3.7.0.tar.gz .
RUN tar -zxf zh_core_web_sm-3.7.0.tar.gz -C /root/model/zh
EXPOSE 80
ADD ./app.py .
ADD ./vllm_utils.py .
ADD ./run_vllm_server.sh .
