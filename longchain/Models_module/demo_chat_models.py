import os
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.messages import HumanMessage
# # 1.设置qianfan的API-KEY和Serect-KEY
# os.environ["QIANFAN_AK"] = QIANFAN_AK
# os.environ["QIANFAN_SK"] = QIANFAN_SK

# 2.实例化模型
chat = QianfanChatEndpoint(model='Qianfan-Chinese-Llama-2-7B')
messages = [HumanMessage(content="给我写一首唐诗")]
res = chat.invoke(messages)
print(res)
