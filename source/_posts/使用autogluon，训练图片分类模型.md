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
    ```python
    %matplotlib inline
    import autogluon.core as ag
    from autogluon.vision import ImagePredictor
    import pandas as pd

    #已将数据集放在/train_valid_test/下，分为train,train_valid和test
    train_dataset = ImagePredictor.Dataset.from_folder('./data/train_valid_test/train_valid/')  
    print(train_dataset)
    model_list = ImagePredictor.list_models()
    #查看ImagePredictor提供的model
    print(model_list)

    #电脑性能不行，自选部分超参数
    #model = ag.Categorical('resnet50_v2', 'mobilenetv3_small')
    batch_size=6
    epochs= 200
    #hyperparameters={'model': model, 'batch_size': batch_size,'epochs':epochs}
    hyperparameters={'batch_size': batch_size,'epochs':epochs}

    #设置时间限制
    time_limit = 72* 60 * 60
    predictor = ImagePredictor()
    predictor.fit(train_dataset, hyperparameters=hyperparameters)
    print('Top-1 val acc: %.3f' % predictor.fit_summary()['valid_acc'])

    test_dataset = ImagePredictor.Dataset.from_folder('./data/train_valid_test/test/')
    pred = predictor.predict(test_dataset)
    print(pred)
    ag.utils.generate_csv(pred.tolist(), './submission.csv')
    label_list=['abies_concolor',
    'abies_nordmanniana',
    'acer_campestre',
    'acer_ginnala',
    'acer_griseum',
    'acer_negundo',
    'acer_palmatum',
    'acer_pensylvanicum',
    'acer_platanoides',
    'acer_pseudoplatanus',
    'acer_rubrum',
    'acer_saccharinum',
    'acer_saccharum',
    'aesculus_flava',
    'aesculus_glabra',
    'aesculus_hippocastamon',
    'aesculus_pavi',
    'ailanthus_altissima',
    'albizia_julibrissin',
    'amelanchier_arborea',
    'amelanchier_canadensis',
    'amelanchier_laevis',
    'asimina_triloba',
    'betula_alleghaniensis',
    'betula_jacqemontii',
    'betula_lenta',
    'betula_nigra',
    'betula_populifolia',
    'broussonettia_papyrifera',
    'carpinus_betulus',
    'carpinus_caroliniana',
    'carya_cordiformis',
    'carya_glabra',
    'carya_ovata',
    'carya_tomentosa',
    'castanea_dentata',
    'catalpa_bignonioides',
    'catalpa_speciosa',
    'cedrus_atlantica',
    'cedrus_deodara',
    'cedrus_libani',
    'celtis_occidentalis',
    'celtis_tenuifolia',
    'cercidiphyllum_japonicum',
    'cercis_canadensis',
    'chamaecyparis_pisifera',
    'chamaecyparis_thyoides',
    'chionanthus_retusus',
    'chionanthus_virginicus',
    'cladrastis_lutea',
    'cornus_florida',
    'cornus_kousa',
    'cornus_mas',
    'crataegus_crus-galli',
    'crataegus_laevigata',
    'crataegus_phaenopyrum',
    'crataegus_pruinosa',
    'crataegus_viridis',
    'cryptomeria_japonica',
    'diospyros_virginiana',
    'eucommia_ulmoides',
    'evodia_daniellii',
    'fagus_grandifolia',
    'ficus_carica',
    'fraxinus_nigra',
    'fraxinus_pennsylvanica',
    'ginkgo_biloba',
    'gleditsia_triacanthos',
    'gymnocladus_dioicus',
    'halesia_tetraptera',
    'ilex_opaca',
    'juglans_cinerea',
    'juglans_nigra',
    'juniperus_virginiana',
    'koelreuteria_paniculata',
    'larix_decidua',
    'liquidambar_styraciflua',
    'liriodendron_tulipifera',
    'maclura_pomifera',
    'magnolia_acuminata',
    'magnolia_denudata',
    'magnolia_grandiflora',
    'magnolia_macrophylla',
    'magnolia_stellata',
    'magnolia_tripetala',
    'magnolia_virginiana',
    'malus_baccata',
    'malus_coronaria',
    'malus_floribunda',
    'malus_hupehensis',
    'malus_pumila',
    'metasequoia_glyptostroboides',
    'morus_alba',
    'morus_rubra',
    'nyssa_sylvatica',
    'ostrya_virginiana',
    'oxydendrum_arboreum',
    'paulownia_tomentosa',
    'phellodendron_amurense',
    'picea_abies',
    'picea_orientalis',
    'picea_pungens',
    'pinus_bungeana',
    'pinus_cembra',
    'pinus_densiflora',
    'pinus_echinata',
    'pinus_flexilis',
    'pinus_koraiensis',
    'pinus_nigra',
    'pinus_parviflora',
    'pinus_peucea',
    'pinus_pungens',
    'pinus_resinosa',
    'pinus_rigida',
    'pinus_strobus',
    'pinus_sylvestris',
    'pinus_taeda',
    'pinus_thunbergii',
    'pinus_virginiana',
    'pinus_wallichiana',
    'platanus_acerifolia',
    'platanus_occidentalis',
    'populus_deltoides',
    'populus_grandidentata',
    'populus_tremuloides',
    'prunus_pensylvanica',
    'prunus_sargentii',
    'prunus_serotina',
    'prunus_serrulata',
    'prunus_subhirtella',
    'prunus_virginiana',
    'prunus_yedoensis',
    'pseudolarix_amabilis',
    'ptelea_trifoliata',
    'pyrus_calleryana',
    'quercus_acutissima',
    'quercus_alba',
    'quercus_bicolor',
    'quercus_cerris',
    'quercus_coccinea',
    'quercus_imbricaria',
    'quercus_macrocarpa',
    'quercus_marilandica',
    'quercus_michauxii',
    'quercus_montana',
    'quercus_muehlenbergii',
    'quercus_nigra',
    'quercus_palustris',
    'quercus_phellos',
    'quercus_robur',
    'quercus_shumardii',
    'quercus_stellata',
    'quercus_velutina',
    'quercus_virginiana',
    'robinia_pseudo-acacia',
    'salix_babylonica',
    'salix_caroliniana',
    'salix_matsudana',
    'salix_nigra',
    'sassafras_albidum',
    'staphylea_trifolia',
    'stewartia_pseudocamellia',
    'styrax_japonica',
    'taxodium_distichum',
    'tilia_americana',
    'tilia_cordata',
    'tilia_europaea',
    'tilia_tomentosa',
    'tsuga_canadensis',
    'ulmus_americana',
    'ulmus_glabra',
    'ulmus_parvifolia',
    'ulmus_procera',
    'ulmus_pumila',
    'ulmus_rubra',
    'zelkova_serrata']

    a = pd.read_csv('./data/test.csv')
    b = pd.read_csv('./submission.csv')

    id = []
    for i in range(len(a)):
        id.append(a['image'][i])

    label=[]
    for i in range(len(a)):
        label.append(str(label_list[int(b['category'][i])]))

    df = pd.DataFrame({'image':id,'label':label})

    df.to_csv("test.csv",index=False,sep=',')
    ```

3. 使用AutoGluon，在valid_dataset上准确率为95%
4. 使用submission.csv上传kaggle

## 整理图片代码：
```python
import collections
import math
import os
import shutil

def read_csv_labels(fname):
    """读取 `fname` 来给标签字典返回一个文件名。"""
    with open(fname, 'r') as f:
        # 跳过文件头行 (列名)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

data_dir='./data'
labels = read_csv_labels(os.path.join(data_dir, 'train.csv'))
print('# 训练示例 :', len(labels))
print('# 类别 :', len(set(labels.values())))

#@save
def copyfile(filename, target_dir):
    """将文件复制到目标目录。"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

#@save
def reorg_train_valid(data_dir, labels, valid_ratio):
    # 训练数据集中示例最少的类别中的示例数
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集中每个类别的示例数
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir ,'images')):
        fname = os.path.join(data_dir, 'images', train_file)
        try:
            label = labels['images/' + train_file]
            copyfile(
                fname,
                os.path.join(data_dir, 'train_valid_test', 'train_valid', label))
            if label not in label_count or label_count[label] < n_valid_per_label:
                copyfile(
                    fname,
                    os.path.join(data_dir, 'train_valid_test', 'valid', label))
                label_count[label] = label_count.get(label, 0) + 1
            else:
                copyfile(
                    fname,
                    os.path.join(data_dir, 'train_valid_test', 'train', label))
        except:
            copyfile(
                fname,
                os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))
    return n_valid_per_label

def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'train.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)

batch_size = 64
valid_ratio = 0.5
reorg_cifar10_data(data_dir, valid_ratio)
```
