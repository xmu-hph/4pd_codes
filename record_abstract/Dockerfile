FROM harbor.4pd.io/lab-platform/pk_platform/model_services/hph_base:jupyter_nvcc_torch_ubuntu2204_base_image_vscode
#FROM nvcr.io/nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
ARG HTTP_PROXY=http://10.100.116.50:7890
ARG HTTPS_PROXY=http://10.100.116.50:7890
ARG all_proxy=socks5://10.100.116.50:7890
RUN pip install websockets wave aiohttp librosa
RUN pip install loguru flask_socketio eventlet fastapi pydantic uvicorn
RUN pip install spacy spacy-pkuseg transformers
EXPOSE 80
RUN mkdir /home/code
WORKDIR /home/code
RUN pip install accelerate
ADD ./app.py .
