from zhipuai import ZhipuAI
from dotenv import load_dotenv, find_dotenv
from sql_function_tools import *
import os


_ = load_dotenv(find_dotenv())

# 获取环境变量 ZhiPu_API_KEY
zhupu_ak = os.environ['ZHIPU_API_KEY']
client = ZhipuAI(api_key=zhupu_ak) # 填写您自己的APIKey
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
    messages.append({"role": "system",
                     "content": "通过针对业务数据库生成 SQL 查询来回答用户的问题"})
    messages.append({"role": "user", "content": "查询一下最低工资的员工姓名及对应的工资"})
    response = chat_completion_request(
        messages, tools=tools, tool_choice="auto"
    )
    assistant_message = response.choices[0].message
    print(f'assistant_message1-->{assistant_message}')
    function_name = response.choices[0].message.tool_calls[0].function.name
    function_id = response.choices[0].message.tool_calls[0].id
    function_response = parse_response(response)
    print(f'assistant_message.model_dump()-->{assistant_message.model_dump()}')
    messages.append(assistant_message.model_dump())  # extend conversation with assistant's reply
    messages.append(
        {
            "role": "tool",
            "tool_call_id": function_id,
            "name": function_name,
            "content": str(function_response),
        }

    )  # extend conversation with function response
    print(f'messages-->{messages}')
    last_response = chat_completion_request(
        messages, tools=tools, tool_choice="auto"
    )
    print(f'last_response--》{last_response}')
    print(f'last_response--》{last_response.choices[0].message}')
if __name__ == '__main__':
    main()