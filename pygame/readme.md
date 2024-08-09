# 服务器中搭建虚拟显示
服务器中一般没有显示器，因此需要安装虚拟显示
```sh
apt install xvfb
Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1
export SC2PATH=~/StarCraftII/
```