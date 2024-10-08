import os
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

# os.environ["QIANFAN_AK"] = QIANFAN_AK
# os.environ["QIANFAN_SK"] = QIANFAN_SK
# 1.读取文档里面的内容：pku.txt

with open('./pku.txt',encoding='utf-8') as f:
    pku_str1 = f.read()
# print(pku_str1)

# 2.要切分文档
text_spliter = CharacterTextSplitter(chunk_size=100, chunk_overlap=5)
texts = text_spliter.split_text(pku_str1)
print(len(texts))
print('*'*80)
# 3.将切分后的文档向量化并保存
embedd = QianfanEmbeddingsEndpoint()
docsearch = Chroma.from_texts(texts, embedd)

query = "1937年北京大学发生了什么？"
result = docsearch.similarity_search(query)
print(result)
print(len(result))
