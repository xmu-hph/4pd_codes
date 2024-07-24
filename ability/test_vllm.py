from openai import OpenAI
from openai import APIConnectionError
import time

client = OpenAI(
    base_url="http://localhost:10086/v1",
    api_key="123456",
)

i = 0
probe_query = "Are you ready?!"
while True:
    try:
        print(probe_query)
        completion = client.chat.completions.create(
          model="llm",
          messages=[
            {"role": "user", "content": probe_query}
          ]
        )
        print(completion.choices[0].message)
        break
    except APIConnectionError:
        i += 1
        print(f"No")
        time.sleep(5)
