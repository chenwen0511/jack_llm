from langchain.tools import tool
import datetime
import os
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

import account_secret

my_126_pwd = os.environ['MY_126_PWD']


class CustomTools():

    @tool("将文本写入文档中")
    def store_poesy_to_txt(content: str) -> str:
        """
        将编辑后的书信文本内容自动保存到txt文档中
        """
        try:

            filename = r"poie.txt"
            # 将文本写入txt文档中
            with open(filename, 'w') as file:
                file.write(content)
            
            # 返回结果展示：文件已经写入
            return f"File written to {filename}."
        except Exception:
            return "Error with the input for the tool."

    @tool("发送文本到邮件")
    def send_message(self):
        '''
        读取生成的本地书信文件txt文本,并以邮件的形式发送到某个人的邮箱中
        :return:
        '''


        # 发件人
        from_name = "小可爱"
        # 发件邮箱
        from_addr = "chenwen0511@126.com"
        # 发件邮箱授权码，注意不是QQ邮箱密码
        from_pwd = my_126_pwd  # 注意这里是授权码 不是邮箱的登录密码
        # 收件邮箱
        to_addr = "chenwen0511@outlook.com"  # 接收邮件的邮箱
        # 邮件标题
        my_title = "520小情书"
        # 邮件正文
        # 书信路径
        filename = r"../poie.txt"

        # 定义md文档的绝对路径
        with open(filename, encoding="utf-8", mode="r")as f:
            my_msg = f.read()
        # MIMEText三个主要参数
        # 1. 邮件内容
        # 2. MIME子类型，plain表示text类型
        # 3. 邮件编码格式，使用"utf-8"避免乱码
        msg = MIMEText(my_msg, 'plain', 'utf-8')
        msg['From'] = formataddr([from_name, from_addr])
        # 邮件的标题
        msg['Subject'] = my_title
        # SMTP服务器地址，QQ邮箱的SMTP地址是"smtp.qq.com"
        # smtp_srv = "smtp.163.com"
        smtp_srv = "smtp.126.com"  # todo 注意 这里需要修改smtp服务器的地址


        try:
            # 不能直接使用smtplib.SMTP来实例化，第三方邮箱会认为它是不安全的而报错
            # 使用加密过的SMTP_SSL来实例化，它负责让服务器做出具体操作，它有两个参数
            # 第一个是服务器地址，但它是bytes格式，所以需要编码
            # 第二个参数是服务器的接受访问端口，SMTP_SSL协议默认端口是465
            srv = smtplib.SMTP_SSL(smtp_srv.encode(), 465)
            # 使用授权码登录QQ邮箱
            srv.login(from_addr, from_pwd)
            # 使用sendmail方法来发送邮件，它有三个参数
            # 第一个是发送地址
            # 第二个是接受地址，是list格式，可以同时发送给多个邮箱
            # 第三个是发送内容，作为字符串发送
            srv.sendmail(from_addr, [to_addr], msg.as_string())
            print('发送成功')
        except Exception as e:
            print('发送失败')
            raise e
        finally:
            # 无论发送成功还是失败都要退出你的QQ邮箱
            srv.quit()

if __name__ == '__main__':
    tool = CustomTools()
    tool.send_message()
