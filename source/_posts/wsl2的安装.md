---
title: wsl2的安装
date: 2021-05-18 11:08:42
tags: 
  - wsl2
description: 
  - wsl2的安装
categories: 
  - Linux
copyright_author: Tensor
copyright_author_href: https://github.com/ZLLVZY
copyright_url: https://zllvzy.github.io/2021/05/18/wsl2%E7%9A%84%E5%AE%89%E8%A3%85/
copyright_info: 此文章版权归Tensor所有，如有转载，请注明来自原作者
cover: https://www.wallpaperup.com/uploads/wallpapers/2014/02/19/259727/f4056cdb600dc71619927518abfa3350.jpg

---
# wsl2的安装
[微软官方教程](https://docs.microsoft.com/en-us/windows/wsl/wsl2-install)

教程大抵如下

1. 控制面板 添加或关闭windows功能 确认 <适用于linux的windows子系统> <虚拟机平台> 两项功能支持被添加

2. 重启电脑 使更改生效

3. 更改 wsl默认版本 `wsl --set-default-version 2`

4. 从商店中安装发行版安装包，或者 `wsl --set-version <Distro> 2` 对已有发行进行版本升级
