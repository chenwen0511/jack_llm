import os
from langchain_community.chat_models import QianfanChatEndpoint
from langchain.chains import ConversationChain

# 1.设置qianfan的API-KEY和Serect-KEY
# os.environ["QIANFAN_AK"] = QIANFAN_AK
# os.environ["QIANFAN_SK"] = QIANFAN_SK

# 2.实例化模型
llm = QianfanChatEndpoint(model="Qianfan-Chinese-Llama-2-7B")

# 3. 实例化Chain
conver_chain = ConversationChain(llm=llm)

result1 = conver_chain.predict(input="小明有1只猫")
print(f'result1--->{result1}')
result2 =conver_chain.predict(input="小刚有2只狗")
print(f'result2--->{result2}')
result3 =conver_chain.predict(input="小明和小刚一共有多少只宠物")
print(f'result3--->{result3}')
