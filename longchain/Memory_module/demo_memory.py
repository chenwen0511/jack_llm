from langchain.memory import ChatMessageHistory

# 1.实例对象
history = ChatMessageHistory()
# 2. 添加历史信息
history.add_user_message("在吗")
history.add_ai_message("有什么事吗？")

print(history.messages)