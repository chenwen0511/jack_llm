import json
import requests
import os
import pymysql
from dotenv import load_dotenv, find_dotenv

# todo: 1.描述数据库表结构（单一个表格）
database_schema_string = """
  CREATE TABLE `emp` (
  `empno` int DEFAULT NULL, --员工编号, 默认为空
  `ename` varchar(50) DEFAULT NULL, --员工姓名, 默认为空
  `job` varchar(50) DEFAULT NULL,--员工工作, 默认为空
  `mgr` int DEFAULT NULL,--员工领导, 默认为空
  `hiredate` date DEFAULT NULL,--员工入职日期, 默认为空
  `sal` int DEFAULT NULL,--员工的月薪, 默认为空
  `comm` int DEFAULT NULL,--员工年终奖, 默认为空
  `deptno` int DEFAULT NULL,--员工部分编号, 默认为空
)"""

# todo: 2.描述数据库表结构（多个表格）
database_schema_string1 = """
CREATE TABLE `emp` (
  `empno` int DEFAULT NULL, --员工编号, 默认为空
  `ename` varchar(50) DEFAULT NULL, --员工姓名, 默认为空
  `job` varchar(50) DEFAULT NULL,--员工工作, 默认为空
  `mgr` int DEFAULT NULL,--员工领导, 默认为空
  `hiredate` date DEFAULT NULL,--员工入职日期, 默认为空
  `sal` int DEFAULT NULL,--员工的月薪, 默认为空
  `comm` int DEFAULT NULL,--员工年终奖, 默认为空
  `deptno` int DEFAULT NULL,--员工部分编号, 默认为空
);

CREATE TABLE `DEPT` (
  `DEPTNO` int NOT NULL, -- 部门编码, 默认为空
  `DNAME` varchar(14) DEFAULT NULL,--部门名称, 默认为空
  `LOC` varchar(13) DEFAULT NULL,--地点, 默认为空
  PRIMARY KEY (`DEPTNO`)
);
"""


tools = [
    {
        "type": "function",
        "function": {
            "name": "ask_database",
            "description": "使用此函数回答业务问题，要求输出是一个SQL查询语句",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": f"SQL查询提取信息以回答用户的问题。"
                                       f"SQL应该使用以下数据库模式编写:{database_schema_string1}"
                                       f"查询应该以纯文本返回，而不是JSON。"
                                       f"查询应该只包含MySQL支持的语法。",
                    }
                },
                "required": ["query"],
            },
        }
    }
]


# todo:1.连接数据库，进行sql语句的查询


def ask_database(query):
    """连接数据库，进行查询"""
    # 1.连接到 MySQL 数据库
    print("进入函数内部")
    conn = pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='123456',
        database='itcast',
        charset='utf8mb4',  # 指定游标类，返回结果为字典
    )
    # 2. 创建游标
    cursor = conn.cursor()
    print(f'开始测试')
    # 3. 执行sql语句测试
    # 示例：执行 SQL 查询
    # sql = "SELECT * FROM emp"
    print(f'query--》{query}')
    cursor.execute(query)
    # 4. 获取查询结果
    result = cursor.fetchall()
    # 5.关闭游标
    cursor.close()
    # 6.关闭连接
    conn.close()
    return result


# # todo: 2.根据模型回复来确定使用工具函数：


def parse_response(response):
    response_message = response.choices[0].message
    # 检测是否需要调用函数
    if response_message.tool_calls:
        # 调用函数
        available_functions = {
            "ask_database": ask_database
        }  # only one function test in this example, but you can have multiple
        function_name = response_message.tool_calls[0].function.name
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message.tool_calls[0].function.arguments)
        function_response = fuction_to_call(
            query=function_args.get("query"),
        )
        return function_response


if __name__ == '__main__':
    query = "select count(*) from emp"
    a = ask_database(query)
    print(a)
