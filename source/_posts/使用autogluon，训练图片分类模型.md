---
title: 使用AutoGluon，训练图片分类模型
date: 2021-06-25 15:58:05
tags:
  - AI
  - AutoGluon
categories: 
  - AI
copyright_author: Tensor
copyright_author_href: https://github.com/ZLLVZY
copyright_url: https://zllvzy.github.io/2021/06/25/%E4%BD%BF%E7%94%A8autogluon%EF%BC%8C%E8%AE%AD%E7%BB%83%E5%9B%BE%E7%89%87%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B/
copyright_info: 此文章版权归Tensor所有，如有转载，请注明来自原作者
cover: https://auto.gluon.ai/stable/_static/AutogluonLogo.png

---
# 使用AutoGluon，训练图片分类模型
此次主要使用AutoGluon，数据集为李沐老师在[动手学习深度学习](https://courses.d2l.ai/zh-v2/)课程中卷积神经网络章节中的kaggle竞赛作业[classify-leaves](https://www.kaggle.com/c/classify-leaves)

## 主要步骤如下：
1. 使用autogluon的[docker镜像](https://registry.hub.docker.com/r/autogluon/autogluon/tags?page=1&ordering=last_updated)搭建环境
    - 下载[数据集](https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/29193/2318453/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1624868108&Signature=QZT1LdYifGdDeQI5mL29peYD8EC3jvWb4sgwOALMsowdp6jFfnANl3lnOov7g9fWM8m6qEAx7BnabluiuIi7Ypop7ZJX9lKhQl6npzXh32xAzLbflTZMxBXkqOcxUUqO7TrUTIqAewG6fhe3yAOv%2BAaYNtGeYw3BsZbXbqKNWR3SjqcObxFeJWzqTnULfl5Cc3C3C29n6lMddOXzCyaHLLTCFTuSurxdVSuMu9r4txdTk9iPq8JgCNy5oMK%2BplHOs5VUMiiRmSj%2BbVU%2BJ5eBmwn7arczZpVtv2IbyBGiwBoizVAC2t0rgLjDiW0LxF2Hd%2B76IhJOMwbdw1gclb88AA%3D%3D&response-content-disposition=attachment%3B+filename%3Dclassify-leaves.zip)至本地，此处放在`/home/x/ENV/pytorch`下
    - 使用AutoGluon docker容器  
    `sudo docker run --runtime=nvidia -it --name='pytorch' -p 8888:8888 -e LANG="C.UTF-8" -v /home/x/ENV/pytorch:/workspace/mycode autogluon/autogluon:0.2.0-rapids0.19-cuda10.2-jupyter-ubuntu18.04-py3.7 /bin/bash`
    - 安装vim(编辑jupyter配置文件)  
    `apt-get install vim`
    - 配置jupyter远程访问
    ```
    jupyter notebook --generate-config
    vim ~/.jupyter/jupyter_notebook_config.py
    c.NotebookApp.ip='*'
    ```
    - 启用jupyter_notebook
    `jupyter notebook --allow-root`
2. 具体训练模型代码如下：

3. 使用AutoGluon，在valid_dataset上准确率为95%
4. 使用submission.csv上传kaggle