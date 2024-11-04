### 1. 使用python3.9创建虚拟环境
python -m venv .venv
### 2. 激活虚拟环境
source .venv/bin/activate
### 3. 安装依赖
pip install -r requirements.txt
### 4. 下载ChatGLM
git clone https://www.modelscope.cn/ZhipuAI/chatglm2-6b.git
### 5. 资源要求
操作系统: CentOS 7
CPUs: 8 core(s)，内存： 48G
GPUs: 1卡， A800， 80GB GPUs 
Python: 3.9
Pytorch: 1.11.0
Cuda: 11.3.1

### 6. 修改 glm_confg.py
模型路径 数据路径 checkpoint路径

