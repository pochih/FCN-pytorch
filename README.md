[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source-150x25.png?v=103)](https://github.com/ellerbrock/open-source-badges/)

## 🚘 The easiest implementation of fully convolutional networks

- Task: __semantic segmentation__, it's a very important task for automated driving

- The model is based on CVPR2015 best paper [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)

## Results
### Trials
<img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='result/trials.png' padding='5px' height="150px"></img>

### Predictions
<img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='result/result.gif' padding='5px' height="150px"></img>


## Performance

I train with two popular benchmark dataset: CamVid and Cityscapes

|dataset|n_class|pixel accuracy|
|---|---|---
|Cityscapes|20|96%
|CamVid|32|93%

## Training

### Install packages
```bash
pip3 install -r requirements.txt
```

and download pytorch 0.2.0 from [pytorch.org](pytorch.org)

and download Cityscapes dataset (recommended) or CamVid dataset

### Run the code
- default dataset is CityScapes
```python
python3 python/CityScapes_utils.py 
python3 python/train.py
```

- or train with CamVid
```python
python3 python/CamVid_utils.py 
python3 python/train.py
```

## Author
Po-Chih Huang / [@brianhuang1019](http://brianhuang1019.github.io/)
