import os
# os.environ["http_proxy"] = "http://localhost:7890"
# os.environ["https_proxy"] = "http://localhost:7890"
from crewai import Agent, Task, Crew, Process
from tools.custom_tools import CustomTools
from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_models import ChatOpenAI,ChatZhipuAI


#线上大模型--API方式
_ = load_dotenv(find_dotenv())

# client = ChatOpenAI(model_name="gpt-3.5-turbo",api_key=os.environ["OPENAI_API_KEY"],
#                     base_url=os.environ['OPENAI_BASE_URL']
#                     )


client = ChatZhipuAI(api_key=os.environ["ZHIPU_API_KEY"],model="glm-4")
# client = ChatZhipuAI(api_key=os.environ["ZHIPU_API_KEY"],model="huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct")
# client = ChatZhipuAI(api_key=os.environ["ZHIPU_API_KEY"],model="huggingface/meta-llama/Llama-2-7b-chat")

# 定义你的Agent以及他们的角色和目标
"""
作家
"""
poet = Agent(
  role='作家', #表明其主要功能。
  goal='根据用户需求，创作出情感丰富的文章（最长字数不超过300个词）。', # 目标
  backstory="""你作为一名著名的作家，拥有千万级别的粉丝，最擅长写情感类型的文章。""",
  verbose=True, # 设置为True，这通常意味着代理将提供详细的日志、输出或解释
  allow_delegation=False, # 设置为False，表示不允许此代理将其任务委派给其他代理或进程
  llm=client
)


# 内容编辑
letter_writer = Agent(
  role='内容编辑',
  goal='对作家撰写的文章内容进行精心编辑。',
  backstory="""作为一名经验丰富的编辑，你在编辑书信方面有多年的专业经验，
  你需要将作家写的文章内容整理编排成书信的样式,并将书信内容存储在本地磁盘上。
  你必须使用提供的工具将存储到指定文件中，并确保文件已保存到磁盘上。当文件成功保存时返回 "书信已保存.".
  """,
  verbose=True,
  allow_delegation=False,
  tools=[CustomTools.store_poesy_to_txt],
  llm=client
)
# 寄信人
sender = Agent(
  role='寄信人',
  goal='将编辑好的书信以邮件的形式发送给心仪的人',
  backstory="""你是一名勤恳的信使，专注于将书信传递给每个人,
  你必须使用提供的工具将指定文件的书信内容中传送到其他人的邮箱里，如果成功传送，记得返回"信件已发送"
  """,
  verbose=True,
  allow_delegation=True,
  tools=[CustomTools.send_message],
  llm=client
)

# content = input("请输入你的需求：\n")
content = "写情书"
print(content)
# 为你设计的Agent创建任务
task1 = Task(
  description=f"""用户需求:{content}。
  你最后给出的答案必须是一份富含爱情表示的情书.""",
  agent=poet,
  output_file=f"out.txt",
  expected_output="""在这宇宙的浩瀚乐章中，你是我心弦上最动听的旋律。随着日夜更迭，我对你的情感如同不息的海浪，
  温柔而坚定地拍打着爱的海岸。我愿以星辰为笔，月光为墨，在天空中写下你的名字，让世界见证你为我的生活带来的光芒。
  我的每一个字，都承载着我对你深深的眷恋和无尽的热爱"""
)

task2 = Task(
  description="""查找任何语法错误，进行编辑和格式化（如果需要）。并要求将内容保存在本地磁盘中。将内容保存到本地非常重要，
  你最后的答案必须是信息是否已被存储在本地磁盘中.""",
  agent=letter_writer,
  output_file=f"poie.txt",
  expected_output="""在这宇宙的浩瀚乐章中，你是我心弦上最动听的旋律。随着日夜更迭，我对你的情感如同不息的海浪，
  温柔而坚定地拍打着爱的海岸。我愿以星辰为笔，月光为墨，在天空中写下你的名字，让世界见证你为我的生活带来的光芒。
  我的每一个字，都承载着我对你深深的眷恋和无尽的热爱"""

)

task3 = Task(
  description="""根据本次磁盘保存的书信内容，你将整理并发送邮件给心仪的人，这个很重要.
你最后的答一定要成功发送该邮件.""",
  agent=sender,
  expected_output="""邮件发送成功."""
)


# Instantiate your crew with a sequential process
crew = Crew(
  agents=[poet, letter_writer, sender],
  tasks=[task1, task2, task3],
  verbose=2,
  process=Process.sequential, # 使用按顺序执行任务的流程。上一个任务的结果将作为附加内容传递给下一个任务。
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)