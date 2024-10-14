from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # 向量数据库
# from langchain.document_loaders import UnstructuredFileLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS  # 向量数据库

def main():
    # 定义向量模型路径
    EMBEDDING_MODEL = 'm3e-base'

    # 第一步：加载文档：
    loader = UnstructuredFileLoader('物流信息.txt')
    data = loader.load()
    # print(f'data-->{data}')
    # 第二步：切分文档：
    text_split = RecursiveCharacterTextSplitter(chunk_size=128,
                                                chunk_overlap=4)
    split_data = text_split.split_documents(data)
    # print(f'split_data-->{split_data}')

    # 第三步：初始化huggingface模型embedding
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 第四步：将切分后的文档进行向量化，并且存储下来
    db = FAISS.from_documents(split_data, embeddings)
    db.save_local('./faiss/camp')

    return split_data


if __name__ == '__main__':
    split_data = main()
    print(f'split_data-->{split_data}')

"""
split_data-->[Document(page_content='物流公司：速达物流\n\n公司总部：北京市\n\n业务范围：国际快递、仓储管理\n\n货物追踪：\n\n货物编号：ABC123456\n\n发货日期：2024\n\n06\n\n15\n\n当前位置：上海分拨中心\n\n预计到达日期：2024\n\n06\n\n20\n\n运输方式：\n\n运输公司：快运通', metadata={'source': '物流信息.txt'}), Document(page_content='运输方式：陆运\n\n出发地：广州\n\n目的地：重庆\n\n预计运输时间：3天\n\n仓储信息：\n\n仓库名称：东方仓储中心\n\n仓库位置：深圳市\n\n存储货物类型：电子产品\n\n存储条件：常温仓储\n\n当前库存量：1000件', metadata={'source': '物流信息.txt'})]


"""