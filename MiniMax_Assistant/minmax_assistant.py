import requests
import json
import time
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

GroupId = os.environ['MM_GROUPID']
API_KEY = os.environ['MM_API_KEY']

headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}
headers_retrieval = {
    'Authorization': f'Bearer {API_KEY}',
    'authority': 'api.minimax.chat',
}


# 流程零：上传文档
def create_file():
    url = f"https://api.minimax.chat/v1/files/upload?GroupId={GroupId}"
    data = {
        'purpose': 'assistants'
    }

    files = {
        'file': open(r'fruit_price.txt', 'rb')
    }
    response = requests.post(url, headers=headers_retrieval, data=data, files=files)
    return response.json()

#流程一：创建助手
def create_assistant(file_id):
    url = f"https://api.minimax.chat/v1/assistants/create?GroupId={GroupId}"
    payload = json.dumps({
        "model": "abab5.5-chat",# 模型版本
        "name": "水果店财务助手", # 助手名称
        "description": "水果店财务助手，用在水果销售过程中计算营业额", # 助手描述
        "instructions": "是一个理财能手，根据每类水果的售出量以及单价，统计其成本和收入，计算出总利润",# 助手设定（即bot_setting)
        "file_ids": [
            str(file_id)
        ],
        "tools": [{"type": "retrieval"}]

    })
    response = requests.post(url, headers=headers, data=payload)
    return response.json()

# 流程二：创建线程
def create_thread():
    url = f"https://api.minimax.chat/v1/threads/create?GroupId={GroupId}"
    response = requests.post( url, headers=headers)
    return response.json()

# 流程三：往线程里添加信息
def add_message_to_thread(thread_id):
    url = f"https://api.minimax.chat/v1/threads/messages/add?GroupId={GroupId}"
    payload = json.dumps({
        "thread_id": thread_id,
        "role": "user",
        "content": "我卖了2斤葡萄，3斤半的香蕉，2斤苹果，计算下总成本和总收入，给出具体的计算过程",
    })
    response = requests.post(url, headers=headers, data=payload)
    return response.json()

# 流程四：运行助手
def run_thread_with_assistant(thread_id, assistant_id):
    # time.sleep(20) #创建assistants进行向量化以及存储时需要一定的时间，可以考虑使用retrieve assistant检索是否创建成功
    url = f"https://api.minimax.chat/v1/threads/run/create?GroupId={GroupId}"
    payload = json.dumps({
        "thread_id": thread_id,
        "assistant_id": assistant_id
    })
    response = requests.post(url, headers=headers, data=payload)
    return response.json()

# 流程五：查看回复内容
def get_thread_messages(thread_id):
    url = f"https://api.minimax.chat/v1/threads/messages/list?GroupId={GroupId}"
    payload = json.dumps({
        "thread_id": thread_id
    })
    response = requests.get(url, headers=headers, data=payload)
    return response.json()

# 查看运行状态
def check_thread_run_status(thread_id, run_id):
    url = f"https://api.minimax.chat/v1/threads/run/retrieve?GroupId={GroupId}"
    payload = json.dumps({
        "thread_id": str(thread_id),
        "run_id": str(run_id)
    })
    completed = False
    while not completed:
        response = requests.request("GET", url, headers=headers, data=payload)
        if response.status_code == 200:
            response_data = response.json()
            status = response_data.get('status', '')
            print(f"Status: {status}")
            if status == 'completed':
                completed = True
                print("Process completed, exiting loop.")
            else:
                time.sleep(2)  # 如果状态不是completed，等待两秒后重新请求
        else:
            print(f"Error: {response.status_code}")
            break  # 如果请求失败，退出循环
    return completed
# 主流程
def main():
    # 上传文档
    file_response = create_file()
    print(f'file_response--》{file_response}')
    file_id = file_response.get('file', {}).get('file_id')
    print("file_id:",file_id)

    # 创建助手
    assistant_response = create_assistant(file_id)
    print(f'assistant_response-->{assistant_response}')
    assistant_id = assistant_response.get('id', '')
    print("assistant_id:",assistant_id)

    # 创建线程
    thread_response = create_thread()
    thread_id = thread_response.get('id', '')
    print("thread_id:",thread_id)

    # 往线程里添加信息
    add_message_to_thread(thread_id)  # 不保存返回值

    # 使用助手处理该线程
    run_response = run_thread_with_assistant(thread_id, assistant_id)
    run_id = run_response.get('id', '')  # 假设run_response是正确的JSON响应，并包含run_id
    print("run_id:", run_id)

    # 检查助手处理状态
    if check_thread_run_status(thread_id, run_id):
        # 获取线程中助手处理出的新信息
        thread_messages_response = get_thread_messages(thread_id)
        print(type(thread_messages_response))
        print(thread_messages_response["data"][1]["content"][0]["text"])
        # 打印JSON数据
        print(json.dumps(thread_messages_response, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()