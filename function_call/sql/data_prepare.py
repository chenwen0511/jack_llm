import sqlite3
from sql_function_tools import database_schema_string


conn = sqlite3.connect('example.db')

conn.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT UNIQUE
)
''')

conn.execute("""
CREATE TABLE IF NOT EXISTS emp (
  empno int DEFAULT NULL, 
  ename varchar(50) DEFAULT NULL, 
  job varchar(50) DEFAULT NULL,
  mgr int DEFAULT NULL,
  hiredate date DEFAULT NULL,
  sal int DEFAULT NULL,
  comm int DEFAULT NULL,
  deptno int DEFAULT NULL
)
""")

for i in range(100):
    conn.execute("INSERT INTO emp (empno, ename, job, mgr, hiredate, sal, comm, deptno) VALUES ({}, '张三', 'Manager', NULL, '2024-01-01', {}, 1000, 10);".format(i+1, i+1))
conn.commit()


ret = conn.execute("select * from emp")
print(ret.fetchall())

print(conn.execute("SELECT ename, sal FROM emp WHERE sal = (SELECT MIN(sal) FROM emp)").fetchall())

conn.close()


