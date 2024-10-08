import os
from langchain_community.llms import QianfanLLMEndpoint
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain_core.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory


# os.environ["QIANFAN_AK"] = QIANFAN_AK
# os.environ["QIANFAN_SK"] = QIANFAN_SK

# 实例化模型
llm = QianfanLLMEndpoint(model="Qianfan-BLOOMZ-7B-compressed")



# 定义工具:这里指定两个工具来选择使用：llm-math计算，wikipedia
tools = load_tools(["llm-math", "wikipedia"], llm=llm)


# 实例化agent
agent = initialize_agent(tools=tools,
                         llm=llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=True)

print(f'agent-->{agent}')

from langchain import PromptTemplate
prompt_template = "中国目前有多少人口"
prompt = PromptTemplate.from_template(prompt_template)
print('prompt-->', prompt)
res = agent.run(prompt)
print(res)