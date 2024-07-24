from openai import OpenAI
from openai import APIConnectionError
import time
import subprocess
from subprocess import DEVNULL
import logging

def run_vllm_server(probe_gap=5):
    p = subprocess.Popen(
        ["sh", "run_vllm_server.sh"],
        stdout=DEVNULL,
        stderr=DEVNULL
    )
    
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
            return client
        except APIConnectionError:
            i += 1
            logging.info(f"Probe readiness: try {i}")
            time.sleep(probe_gap)
