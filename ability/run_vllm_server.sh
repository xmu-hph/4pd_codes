unset http_proxy
unset https_proxy
#model_id=/root/Qwen2-7B-Instruct
model_id=/root/model/Meta-Llama-3.1-8B-Instruct
model_name_ext=llm
python -m vllm.entrypoints.openai.api_server --model ${model_id}  --served-model-name ${model_name_ext} --trust-remote-code --port 10086 --api-key 123456 --max-model-len 4096