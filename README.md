# [FATCNet : Feature Adaptive Transformer and CNN for Infrared Small Target Detection]()


The directory structure of the whole project is as follows:
```bash
.
├── FATCNet
│   ├──datasets_n
│   │       └── dataset_*.py
│   ├──train.py
│   ├──test.py
│   └──...
│   ├── model
│      └── vit_checkpoint
│         └── imagenet21k
│             └── R50+ViT-B_16.npz
│      └── NUAA
│         └── epoch_best.pth
│      └── ...
│   ├── networks
│      ├──vit_seg_modeling.py
│      └──...
│   ├── util
│      └──...

```

The weight file "R50+ViT-B16.npz" required for the project in the project model folder, as well as the weight files required for inference on various datasets and test sets, can be downloaded from the following link:

https://www.dropbox.com/scl/fo/wilsvqswms5i2qlh4lujt/AAwNIMxrsmJkB2JZD6CV4Y4?rlkey=rg4wyr1w43mzunbum4o8bzulv&e=1&st=seavpsn0&dl=0

(PS: We've found that some people are getting errors when running projects using the weights in the ‘model’ folder of their git project. So if you get an error, you can download a new weights file via the link to replace it).

## Installation
```angular2html
pip install -r requirement.txt
```

## Training

```
python train.py
```

## Test

```
python test.py
```
