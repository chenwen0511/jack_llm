import os
from langchain_community.llms import QianfanLLMEndpoint
from langchain_core.prompts import PromptTemplate

# os.environ["QIANFAN_AK"] = QIANFAN_AK
# os.environ["QIANFAN_SK"] = QIANFAN_SK
#1. 定义模板
template = "我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字"
prompt = PromptTemplate(input_variables=["lastname"], template=template)
prompt_text = prompt.format(lastname="李")
print(f'prompt_text-->{prompt_text}')

# 2.实例化模型
llm = QianfanLLMEndpoint(model="Qianfan-Chinese-Llama-2-7B")
# 3.送入模型prompt
result = llm.invoke(prompt_text)
print(f'result--》{result}')
