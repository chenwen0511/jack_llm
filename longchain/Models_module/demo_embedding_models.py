import os
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
# 1.设置qianfan的API-KEY和Serect-KEY
# os.environ["QIANFAN_AK"]=***
# os.environ["QIANFAN_SK"]=***
# 2.实例化向量模型
embed = QianfanEmbeddingsEndpoint(model='Embedding-V1')

# 3.一段文本的向量化
result1 = embed.embed_query("这是一个测试文档")
print(f'result1-->{result1}')
print(f'result1的长度-->{len(result1)}')

# 4.一批文本向量化
result2 = embed.embed_documents(["这是一个测试文档", "这是二个测试文档"])
print(result2)
print(len(result2))
print(len(result2[0]))
print(len(result2[1]))
