### 1. 本地部署

下载代码 
git clone https://github.com/QwenLM/Qwen.git

创建虚拟环境
python -m venv venv

激活虚拟环境
source venv/bin/activate

安装requirements
pip install -r requirements.txt

安装web运行的requirements
pip install -r requirements_web_demo.txt


下载模型
git clone https://www.modelscope.cn/Qwen/Qwen-1_8B-Chat.git

修改web_demo.py中模型的位置

启动服务
python web_demo.py --server-name 127.0.0.1 -c Qwen2-0.5B-Instruct --cpu-only
注：GPU 去掉 --cpu-only  -c 为模型的路径

浏览器访问：http://127.0.0.1:8000/



### 2. 微调