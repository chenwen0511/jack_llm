import os

from langchain_community.chat_models import ChatZhipuAI
import account_secret


from langchain import PromptTemplate, LLMChain
from langchain_community.chat_models import ChatZhipuAI

def generate_love_letter(api_key):
    try:
        # 创建 ChatZhipuAI 实例
        chat_model = ChatZhipuAI(api_key=api_key)

        # 定义提示模板
        template = """
        你是一位浪漫的诗人，擅长写情书。
        请帮我写一封表达爱意的情书。
        """

        # 创建 PromptTemplate 对象
        prompt = PromptTemplate(template=template, input_variables=[])

        # 创建 LLMChain 对象
        chain = LLMChain(llm=chat_model, prompt=prompt)

        # 调用模型生成情书
        response = chain.run({})

        # 返回情书内容
        return response
    except Exception as e:
        print("Unable to generate love letter.")
        print(f"Exception: {e}")
        return str(e)

# 示例使用
api_key = os.environ.get("ZHIPU_API_KEY")
love_letter = generate_love_letter(api_key)
print(love_letter)