from zhipuai import ZhipuAI
# from dotenv import load_dotenv, find_dotenv
from muti_utils import *
from airplane_function_tools import *
import os

# 加载账户密码信息
import account_secret


# _ = load_dotenv(find_dotenv())
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
                     "content": "现在你是一个航班查询助手，将根据用户问题提供答案，但是不要假设或猜测传入函数的参数值。如果用户的描述不明确，请要求用户提供必要信息 "})
    messages.append({"role": "user", "content": "帮我查询2024年6月25日，郑州到北京的航班的票价"})
    # 1.得到第一次回复：调用：get_plane_number函数
    first_response = chat_completion_request(
        messages, tools=tools, tool_choice="auto")
    print(first_response)
    assistant_message1 = first_response.choices[0].message
    print(f'assistant_message1-->{assistant_message1}')
    # 2. 将第一次得到的模型回复结果加入messages
    messages.append(first_response.choices[0].message.model_dump())
    # 3. 第一次得到函数的结果
    first_function = parse_function_call(model_response=first_response)
    print(f'first_function--》{first_function}')

    tool_call = first_response.choices[0].message.tool_calls[0]
    # 4. 将函数的结果添加到messages中，继续送入模型问答

    messages.append({"role": "tool",
                     "tool_call_id": tool_call.id,
                     "content": str(json.dumps(first_function))})
    # 5. 第二次调用模型
    print(messages)
    second_response = chat_completion_request(
        messages, tools=tools, tool_choice="auto")
    print(f'second_response--》{second_response.choices[0].message}')
    # 6. 将第二次得到函数结果加入信息中
    messages.append(second_response.choices[0].message.model_dump())
    second_function = parse_function_call(model_response=second_response)
    print(f'second_function--》{second_function}')
    tool2_call = second_response.choices[0].message.tool_calls[0]
    # 4. 将函数的结果添加到messages中，继续送入模型问答
    messages.append({"role": "tool",
                     "tool_call_id": tool2_call.id,
                     "content": str(json.dumps(second_function))})

    last_response = chat_completion_request(
        messages, tools=tools, tool_choice="auto")
    print(f'last_response--》{last_response.choices[0].message}')
if __name__ == '__main__':
    main()