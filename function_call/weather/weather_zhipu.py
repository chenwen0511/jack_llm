import os
# from dotenv import load_dotenv, find_dotenv
from tools import *
from zhipuai import ZhipuAI
# _ = load_dotenv(find_dotenv())

import account_secret
zhupu_ak = os.environ.get('ZHIPU_API_KEY') # 避免泄露keys

client = ZhipuAI(api_key=zhupu_ak) # 填写您自己的APIKeypi
ChatGLM = "glm-4"
def chat_completion_request(messages, tools=None, tool_choice=None, model=ChatGLM):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )

        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


def main():
    messages = []
    # messages.append({"role": "system",
    #                  "content": "你是一个天气播报小助手，你需要根据用户提供的地址来回答当地的天气情况，如果用户提供的问题具有不确定性，不要自己编造内容，提示用户明确输入"})
    # messages.append({"role": "user", "content": "今天北京的天气如何"})
    messages.append({"role": "system", "content": "你是一个比较数字大小的高手，根据用户的需要，比较两个数字的大小，如果用户提供的问题具有不确定性，不要自己编造内容，提示用户明确输入"})
    # messages.append({"role": "user", "content": "9.99 和 9.9999 这两个数字哪个更大？"})
    messages.append({"role": "user", "content": "9.99是不是比9.9999大"})
    print(messages)
    response = chat_completion_request(
        messages, tools=tools, tool_choice="auto"
    )
    # 获取函数的结果
    function_response = parse_response(response)
    # 添加第一次模型得到的结果
    assistant_message = response.choices[0].message

    print(f'assistant_message-->{assistant_message}')
    messages.append(assistant_message.model_dump())  # extend conversation with assistant's reply
    function_name = response.choices[0].message.tool_calls[0].function.name
    print(f'function_name--》{function_name}')
    function_id = response.choices[0].message.tool_calls[0].id
    print(f'function_id--》{function_id}')
    # 添加函数返回的结果
    messages.append(
        {
            "role": "tool",
            "tool_call_id": function_id,
            "name": function_name,
            "content": function_response,
        }

    )  # extend conversation with function response
    last_response = chat_completion_request(
        messages, tools=tools, tool_choice="auto"
    )
    print(f'last_response--》{last_response.choices[0].message}')


if __name__ == '__main__':
    main()

