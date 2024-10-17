import time
import os
from openai import OpenAI

def call_gpt(llm_model_name,message,max_retries=10):
    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        base_url=os.environ['OPENAI_API_BASE'],
    )
    retries = 0
    while retries < max_retries:
        try:
            completion = client.chat.completions.create(
                model=llm_model_name,
                messages=[{"role": "user", "content": message}],
            )
            response = completion.choices[0].message.content
            retries+=1
            return response
        except KeyboardInterrupt:
            print("Operation canceled by user.")
            return ''
        except Exception as e:
            print(f"Someting wrong:{e}. Retrying in 1 minute...")
            time.sleep(60) # 等待1分钟
            retries += 1

    print("Max retries reached. Unable to get a response.")
    return None

if __name__=="__main__":
    # test
    os.environ["OPENAI_API_KEY"]="your_openai_api_key"
    os.environ["OPENAI_API_BASE"]="https://api.openai.com/v1"
    output=call_gpt(llm_model_name="gpt-4o-0513",
                    message="hello,introduce yourself")
    print(output)
