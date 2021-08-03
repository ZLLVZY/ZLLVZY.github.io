---
title: Cowboy Outfits Detection Megvii_YOLOX
date: 2021-08-03 14:21:05
tags:
  - AI
  - YOLOX
  - Object Detection
categories: 
  - AI
copyright_author: Tensor
copyright_author_href: https://github.com/ZLLVZY
copyright_url: https://zllvzy.github.io/2021/08/03/Cowboy%20Outfits%20Detection%20Megvii_YOLOX
copyright_info: 此文章版权归Tensor所有，如有转载，请注明来自原作者
cover: https://z3.ax1x.com/2021/08/03/fPjkpq.png 

---
# Cowboy Outfits Detection Megvii_YOLOX

## Overview

-   Install YOLOX

-   Prepare CowboyOutfits datasets in YOLOX /data/datasets

-   Prepare our YOLOX exp & cococlasses

-   Train with pre-trained models

-   Evaluation 

-   Visualize demo

-   Output & Submit

-   Socre

## Competition

此次主要为李沐老师在动手学习深度学习课程中目标检测章节中的 kaggle 竞赛作业[CowBoy Outfits Detection](https://www.kaggle.com/c/cowboyoutfits)，[数据集](https://www.kaggle.com/c/cowboyoutfits/data)见kaggle页面，使用旷视最新的YOLOX实现目标检测。

## YOLOX

![YOLOX](https://z3.ax1x.com/2021/08/03/fPjkpq.png)

[https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

YOLOX is an anchor-free version of YOLO, with a simpler design but better performance! It aims to bridge the gap between research and industrial communities. For more details, please refer to Megvii's [report on Arxiv](https://arxiv.org/abs/2107.08430)

## Install YOLOX

1.   Install YOLOX

```python
%%capture
!git clone https://github.com/Megvii-BaseDetection/YOLOX.git
!pip3 install -r YOLOX/requirements.txt
!cd YOLOX && pip install -e .
```

2.   Install [apex](https://github.com/NVIDIA/apex) & [pycocotools](https://github.com/cocodataset/cocoapi)

```python
%%capture
!git clone https://github.com/NVIDIA/apex
!cd apex && pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
!pip3 install cython
!pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

## Prepare CowboyOutfits datasets in YOLOX /data/datasets

1.    Because the image_id is too large, direct use of the data set will report an error,so replace image_id and create a new mapping of category_id to 1-5 ( Thanks [yyHry](https://www.kaggle.com/herunyu/yolox-for-cowboyoutfits#%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86),this part is using his code )

```python
TypeError: can't convert np.ndarray of type numpy.ulonglong. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
```

```python
import json
from copy import deepcopy
# data_list = ['merged_train.json',]
data_list = ['/kaggle/input/cowboyoutfits/train.json']
# data_list = ['new_valid.json']

cat = {87:1, 1034:5, 131:2, 318:3, 588:4}

dict_list = []
for idx, data in enumerate(data_list):
    with open(data) as f:
        dict_list.append(json.load(f))

new_data = {}
print(dict_list[0].keys())


new_data['info'] = dict_list[0]['info']
# new_data['licenses'] = dict_list[0]['licenses']
new_categories = []
for category in dict_list[0]['categories']:
    new_category = deepcopy(category)
    new_category['id'] = cat[category['id']]
    new_categories.append(new_category)
new_data['categories'] = new_categories
new_data['annotations'] = []
new_data['images'] = []
print(new_data)

anno_count = 1
anno_id_dict = {}
count = 1
anno_dict = {}
for data in dict_list:
    annotations = []
    for annotation in data['annotations']:
        new_annotation = deepcopy(annotation)
        new_annotation['category_id'] = cat[annotation['category_id']]
        if annotation['image_id'] not in anno_dict:
            new_annotation['image_id'] = anno_count
            anno_dict[annotation['image_id']] = anno_count
            anno_count += 1
            anno_id_dict[anno_count] = 1
        else:
            new_annotation['image_id'] = anno_dict[annotation['image_id']]
            anno_id_dict[anno_dict[annotation['image_id']]] += 1
        new_annotation['id'] = count
        count +=1
        annotations.append(new_annotation)
    
    new_data['annotations'] = annotations

    images = []
    
    for image in data['images']:
        new_image = deepcopy(image)
        new_image['id'] = anno_dict[image['id']]
        images.append(new_image)
    new_data['images'] = images

    print(f'annotation size: {len(new_data["annotations"])}')
    print(f'image size: {len(new_data["images"])}')

with open('./train.json', 'w') as res:
    json.dump(new_data, res)
```


2.   Split Train and Valid Data Set  ( Thanks [nekokiku ](https://www.kaggle.com/nekokiku) & [Joshua Z. Zhang](https://www.kaggle.com/zhreshold),this part is using their code )

```python
import random
import copy
import json
from pycocotools.coco import COCO

#def create_subset(c, cats, test_n=180):
def create_subset(c, cats, test_n=2):
    new_coco = {}
    new_coco['info'] = {"description": "CowboySuit",
                        "url": "http://github.com/dmlc/glu|on-cv",
                        "version": "1.0","year": 2021,
                        "contributor": "GluonCV/AutoGluon",
                        "date_created": "2021/07/01"}
    new_coco["licenses"]: [
        {"url": "http://creativecommons.org/licenses/by/2.0/","id": 4,"name": "Attribution License"}]
    cat_ids = c.getCatIds(cats)
    train_img_ids = set()

    test_img_ids = set()
    for cat in cat_ids[::-1]:
        img_ids = copy.copy(c.getImgIds(catIds=[cat]))
        random.shuffle(img_ids)
        tn = min(test_n, int(len(img_ids) * 0.5))
        new_test = set(img_ids[:tn])
        exist_test_ids = new_test.intersection(train_img_ids)
        test_ids = new_test.difference(exist_test_ids)
        train_ids = set(img_ids).difference(test_ids)
        print(tn, len(img_ids), len(new_test), len(test_ids), len(train_ids))
        train_img_ids.update(train_ids)
        test_img_ids.update(test_ids)
        print(len(test_img_ids))

    # prune duplicates
    dup = train_img_ids.intersection(test_img_ids)
    train_img_ids = train_img_ids - dup

    train_anno_ids = set()
    test_anno_ids = set()
    for cat in cat_ids:
        train_anno_ids.update(c.getAnnIds(imgIds=list(train_img_ids), catIds=[cat]))
        test_anno_ids.update(c.getAnnIds(imgIds=list(test_img_ids), catIds=[cat]))

    assert len(train_img_ids.intersection(test_img_ids)) == 0, 'img id conflicts, {} '.format(train_img_ids.intersection(test_img_ids))
    assert len(train_anno_ids.intersection(test_anno_ids)) == 0, 'anno id conflicts'
    print('train img ids #:', len(train_img_ids), 'train anno #:', len(train_anno_ids))
    print('test img ids #:', len(test_img_ids), 'test anno #:', len(test_anno_ids))
    new_coco_test = copy.deepcopy(new_coco)

    new_coco["images"] = c.loadImgs(list(train_img_ids))
    new_coco["annotations"] = c.loadAnns(list(train_anno_ids))
    for ann in new_coco["annotations"]:
        ann.pop('segmentation', None)
    new_coco["categories"] = c.loadCats(cat_ids)

    new_coco_test["images"] = c.loadImgs(list(test_img_ids))
    new_coco_test["annotations"] = c.loadAnns(list(test_anno_ids))
    for ann in new_coco_test["annotations"]:
        ann.pop('segmentation', None)
    new_coco_test["categories"] = c.loadCats(cat_ids)
    print('new train split, images:', len(new_coco["images"]), 'annos:', len(new_coco["annotations"]))
    print('new test split, images:', len(new_coco_test["images"]), 'annos:', len(new_coco_test["annotations"]))
    return new_coco, new_coco_test

coco = COCO('./train.json')

nc, nc_test = create_subset(coco, ['belt', 'sunglasses', 'boot', 'cowboy_hat', 'jacket', ])

with open('./new_train.json', 'w') as f:
    json.dump(nc, f)

with open('./new_valid.json', 'w') as f:
    json.dump(nc_test, f)
```

3.   Prepare CowboyOutfits datasets in YOLOX /data/datasets

```python
!mkdir YOLOX/datasets/COCO
!mkdir YOLOX/datasets/COCO/annotations
```

```python
!cp new_train.json YOLOX/datasets/COCO/annotations/instances_train2017.json
#!cp ../input/cowboy-outfits/train.json YOLOX/datasets/COCO/annotations/instances_train2017.json
!cp new_valid.json YOLOX/datasets/COCO/annotations/instances_val2017.json
!ls YOLOX/datasets/COCO/annotations
```

```python
!ln -s /kaggle/input/cowboy-outfits/images YOLOX/datasets/COCO/train2017
!ln -s /kaggle/input/cowboy-outfits/images YOLOX/datasets/COCO/val2017
```

## Prepare our YOLOX exp & cococlasses

1.   Set your own exp, here use yolox_x

```
!echo "        self.num_classes = 5" >> YOLOX/exps/default/yolox_x.py
!echo "        self.max_epoch = 50" >> YOLOX/exps/default/yolox_x.py
!echo "        self.eval_interval = 1" >> YOLOX/exps/default/yolox_x.py 
!echo "        self.warmup_epochs = 2" >> YOLOX/exps/default/yolox_x.py
#!echo "        self.min_lr_ratio = 0.05" >> YOLOX/exps/default/yolox_x.py
#!echo "        self.data_num_workers = 0" >> YOLOX/exps/default/yolox_x.py

!cat YOLOX/exps/default/yolox_x.py
```

2.   Set coco_classes

```python
!echo "COCO_CLASSES = ('belt','boot','cowboy_hat','jacket','sunglasses')" > "YOLOX/yolox/data/datasets/coco_classes.py"
!cat YOLOX/yolox/data/datasets/coco_classes.py
```

3.   Download pre-trained models

```python
!wget https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_x.pth
!mv yolox_x.pth yolox_x.pth.tar
```

## Train with pre-trained models

`!python YOLOX/tools/train.py -f YOLOX/exps/default/yolox_x.py -d 1 -b 4 --fp16 -o -c yolox_x.pth.tar`

## Evaluation

`!python YOLOX/tools/eval.py -f YOLOX/exps/default/yolox_x.py -c YOLOX_outputs/yolox_x/latest_ckpt.pth -b 4 -d 1 --conf 0.001 --fp16 --fuse`

## Visualize demo

`!python YOLOX/tools/demo.py image -n yolox-x -c YOLOX_outputs/yolox_x/latest_ckpt.pth.tar --path ../input/cowboy-outfits/images/005b9630718c06c7.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu`

![visualize.jpg](https://z3.ax1x.com/2021/08/06/fuoySH.jpg)

## Output & Submit

1.   Define the predict function to return output

```python
def predict(pth,jpg):
    %cd YOLOX
    from yolox.exp import get_exp
    from loguru import logger
    from yolox.utils import fuse_model, get_model_info, postprocess, vis
    from yolox.data.data_augment import preproc
    import torch,cv2
    exp=get_exp('exps/default/yolox_x.py','yolox_x')
    model = exp.get_model()
    #logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.cuda()
    model.eval()
    ckpt_file=pth
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = fuse_model(model)
    img = cv2.imread(jpg)
    img, ratio = preproc(img, exp.test_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # test_size = (640, 640)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.cuda()
    outputs = model(img)
    #outputs = postprocess(outputs, 5, exp.test_conf, exp.nmsthre) #test_conf = 0.01 nmsthre = 0.65
    outputs = postprocess(outputs, 5, 0.25 , 0.45)
    output = outputs[0]
    if output==None:
        %cd ..
        return None,None,None
    output = output.cpu()
    bboxes = output[:, 0:4]
    bboxes=bboxes/ratio
    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]
    %cd ..
    return bboxes,cls,scores
```

```python
pth='/kaggle/working/YOLOX_outputs/yolox_x/latest_ckpt.pth.tar'

# yolox's cls to dataset category_id
categories={0:87,1:131,2:318,3:588,4:1034}
```

2.   Use output and 'valid.csv'/'test.csv' to return submissions

```python
from PIL import Image
def create_submission(df,pth,score_thresh=0.1):
    results = []
    for index, row in df.iterrows():
        img_id = row['id']
        file_name = row['file_name']
        img = Image.open(file_name)
        width, height = img.size
        bboxes,cls,scores=predict(pth,file_name)
        if cls==None:
            continue
        for i, p in enumerate(scores):
            if p> score_thresh:
                roi = bboxes[i]
                pred = {'image_id': img_id,
                        'category_id': categories[int(cls[i])],
                        'bbox': [float(roi[0]), float(roi[1]), float(roi[2]-roi[0]), float(roi[3]-roi[1])],  #yolox bbox is xmin,ymin,xmax,ymax,submission is xmin,ymin,w,h
                        'score': float(p)}
                results.append(pred)
        #print(results)

    return results
```

```python
%%capture
import pandas as pd
import os
root = '/kaggle/input/cowboy-outfits'
submission_df = pd.read_csv('/kaggle/input/cowboy-outfits/valid.csv')  # replace with test.csv on the last day
submission_df['file_name'] = submission_df.apply(lambda x: os.path.join(root, 'images', x['file_name']), axis=1)
submission = create_submission(submission_df, pth)
```

```python
# create json and zip
import zipfile
import json
submission_name = '/kaggle/working/answer.json'
with open(submission_name, 'w') as f:
    json.dump(submission, f)
zf = zipfile.ZipFile('/kaggle/working/sample_answer.zip', 'w')
zf.write(submission_name, 'answer.json')
zf.close()
```

3.   Visualize again (  Thanks [nekokiku ](https://www.kaggle.com/nekokiku) & [snow clem](https://www.kaggle.com/snowclem),this part is using their code )

```python
import pandas as pd
from tqdm import tqdm
import cv2
import json
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from matplotlib.patches import Rectangle 

def get_xyxy_from_cowboy(img_name, df, json_label):
    xy_list = []
    fname_id_dict = {}
    for idx, row in df.iterrows():
        fname_id_dict.update({row['file_name']: row['id']})
    print('len(valid)=', len(fname_id_dict))
    with open(json_label) as f:
        jdata = json.load(f)
        for dict in tqdm(jdata):
            image_id = fname_id_dict[img_name]
            if image_id == dict['image_id']:
                # x_min, y_min, x_max, y_max = dict['bbox']
                x, y, w, h = dict['bbox']
                print(dict['category_id'])
                x_min, y_min, w, h = x, y,w,h
                xy_list.append([int(x_min), int(y_min), int(w), int(h)])

    return xy_list


def draw_rect(img, xy_list):
    for xy in xy_list:
        #cv2.rectangle(img, (xy[0], xy[1]), (xy[2], xy[3]), (0, 0, 255), 2)
        print(xy)
        return Rectangle((xy[0],xy[1]),xy[2], xy[3],fc ='none',ec ='r', lw =2)  


dataset_path = '../input/cowboy-outfits/images'
df = pd.read_csv('../input/cowboy-outfits/valid.csv')
img_name = df['file_name'].sample(1).tolist()[0]
json_label = r'answer.json'

#img_name='d4ab52b2598b8f08.jpg'
print(img_name)
img = cv2.imread(os.path.join(dataset_path, img_name))
print(img.shape)  # (h,w,c)

xy_list = get_xyxy_from_cowboy(img_name, df, json_label)
tmp=draw_rect(img, xy_list)

fig = plt.figure() 
ax = fig.add_subplot() 
plt.imshow(img)
ax.add_patch(tmp) 
```

## Score

![Score.png](https://z3.ax1x.com/2021/08/06/fu4ioF.png)
