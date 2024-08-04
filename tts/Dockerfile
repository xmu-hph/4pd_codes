FROM harbor.4pd.io/lab-platform/pk_platform/model_services/hph_for_4pd_tts:jupyter_nvcc_torch230_cuda121_cudnn8_ubuntu2204_xtts_model_image
#FROM nvcr.io/nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
ARG HTTP_PROXY=http://10.100.116.50:7890
ARG HTTPS_PROXY=http://10.100.116.50:7890
ARG all_proxy=socks5://10.100.116.50:7890
RUN pip install websockets wave aiohttp librosa
EXPOSE 80
EXPOSE 8080
RUN pip install loguru flask_socketio eventlet fastapi pydantic uvicorn
RUN mkdir /home/xtts
WORKDIR /home/xtts
RUN mkdir -p ./examples
ADD ./examples/* ./examples/
RUN pip install spacy spacy-pkuseg
RUN mkdir -p /root/model/tts_1
RUN mkdir -p /root/model/zh_1
RUN mkdir -p /root/model/en_1
COPY en_core_web_sm-3.7.1.tar.gz .
COPY zh_core_web_sm-3.7.0.tar.gz .
RUN tar -zxf en_core_web_sm-3.7.1.tar.gz -C /root/model/en_1
RUN tar -zxf zh_core_web_sm-3.7.0.tar.gz -C /root/model/zh_1
ADD ./app.py .
