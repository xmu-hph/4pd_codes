from openai import OpenAI
from openai import APIConnectionError
import time
import subprocess
from subprocess import DEVNULL
from loguru import logger

def run_vllm_server(probe_gap=5):
    with open("vllm_server.log",'w') as log_file:
        p = subprocess.Popen(
            ["bash", "run_vllm_server.sh"],
            stdout=log_file,
            stderr=log_file
            )
        logger.info(f"加载子进程启动完成")
    client = OpenAI(
        base_url="http://localhost:10086/v1",
        api_key="123456",
    )

    # probe readiness
    i = 0
    probe_query = "Are you ready?!"
    while True:
        try:
            completion = client.chat.completions.create(
              model="llm",
              messages=[
                {"role": "user", "content": probe_query}
              ]
            )
            logger.info(completion)
            logger.info(f"模型加载完成")
            return client
        except APIConnectionError:
            i += 1
            logger.info(f"Probe readiness: try {i}")
            time.sleep(probe_gap)