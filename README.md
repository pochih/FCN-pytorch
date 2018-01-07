[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source-150x25.png?v=103)](https://github.com/ellerbrock/open-source-badges/)

## 🚘 The easiest implementation of fully convolutional networks

- Task: __semantic segmentation__

- It's a very important task for automated driving

- The model is based on CVPR2015 best paper [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)

### performance
I train with two popular benchmark dataset: CamVid and Cityscapes

|dataset|pixel accuracy|
|---|---
|CamVid|> 90%
|Cityscapes|> 90%

### install package
```bash
pip3 install -r requirements.txt
```

and download pytorch 0.2.0 from [pytorch.org](pytorch.org)

and download CamVid dataset or Cityscapes dataset

### training
```python
python3 python/train.py
```

## Author
Po-Chih Huang / [@brianhuang1019](http://brianhuang1019.github.io/)
