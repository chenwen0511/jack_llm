# 加载环境
source .env

# 下载数据
cd data/ || wget https://gitcode.com/open-source-toolkit/94ecd/blob/main/cifar-10-python.tar.gz

# 启动
deepspeed main.py --deepspeed_config config.json

