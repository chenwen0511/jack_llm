import os
from langchain_community.llms import QianfanLLMEndpoint
# 1.设置qianfan的API-KEY和Serect-KEY
# os.environ["QIANFAN_AK"] = QIANFAN_AK
# os.environ["QIANFAN_SK"] = QIANFAN_SK

# 2.实例化模型
llm = QianfanLLMEndpoint(model="ChatGLM2-6B-32K")
res = llm.invoke("请帮我讲一个笑话")
print(res)