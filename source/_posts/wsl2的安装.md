---
title: wsl2的安装
date: 2021-05-18 11:08:42
tags: wsl2
description: wsl2的安装
categories: Linux
cover: https://pics6.baidu.com/feed/a8014c086e061d956d575da485d256d763d9cac2.jpeg?token=11d9515654a985a2bbe48bca1cfabc5f&s=F239716C007AB994DC0ABD180300009B
---
# wsl2的安装
[微软官方教程](https://docs.microsoft.com/en-us/windows/wsl/wsl2-install)

教程大抵如下

1. 控制面板 添加或关闭windows功能 确认 <适用于linux的windows子系统> <虚拟机平台> 两项功能支持被添加

2. 重启电脑 使更改生效

3. 更改 wsl默认版本

wsl --set-default-version 2

4. 从商店中安装发行版安装包，或者 wsl --set-version <Distro> 2 对已有发行进行版本升级
