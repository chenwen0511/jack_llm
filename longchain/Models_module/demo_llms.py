import os
from langchain_community.llms import QianfanLLMEndpoint

import account_secret

# 1.设置qianfan的API-KEY和Serect-KEY
# os.environ["QIANFAN_AK"] = QIANFAN_AK
# os.environ["QIANFAN_SK"] = QIANFAN_SK

# 2.实例化模型
llm = QianfanLLMEndpoint(model="ChatGLM2-6B-32K")
# llm = QianfanLLMEndpoint(model="ERNIE-4.0-8K")
# llm = QianfanLLMEndpoint(model="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/yi_34b_chat")
res = llm.invoke("请帮我讲一个笑话")
print(res)