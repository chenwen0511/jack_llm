[TOC]

[参考1](https://blog.csdn.net/weixin_43822395/article/details/123889195)

## 1. 文件准备

ultraiso工具：
https://www.ultraiso.net/xiazai.html

ubantu 24.04iso镜像
https://ftp.sjtu.edu.cn/ubuntu-cd/24.04/  直接下载会比较慢
https://ftp.sjtu.edu.cn/ubuntu-cd/24.04/ubuntu-24.04.1-desktop-amd64.iso.torrent  # 使用迅雷工具打开这个种子下载会比较快
![](image/Snipaste_2024-09-21_16-30-00.jpg)



# 2. u盘制作-失败案例
注意：准备一个u盘 如果里面有东西请备份到别处

## 2.1 使用ultraiso工具导入ubantu24.04 iso文件
![读入iso](image/a3e7a2a55f000b46579e526679767847.png)

## 2.2 写入u盘
写入方式选择usb+hdd
![写入硬盘](image/9ddd147006d3d224398be6935e21c74f.png)
![写入](image/4fd80167daa68718259f94027da332bf.png)

使用该方法安装遇到报错 error symbol 'grub_calloc' not found, 所以转而更换u盘制作工具。


# 3. u盘制作-rufus
更换制作工具，下载链接以及下载地址：
[rufus](http://rufus.ie/zh/)
[下载链接](https://github.com/pbatard/rufus/releases/download/v4.5/rufus-4.5.exe)

制作截图：
![截图](image/20240923003315.png)


## 3 安装
[参考](https://www.php.cn/faq/775798.html)