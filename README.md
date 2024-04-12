# GCM: Generalized Correspondence Matching via Flexible Hierarchical Refinement and Patch Descriptor Distillation
Python (Pytorch) implementation of our [paper](https://arxiv.org/abs/2403.05388).
<br>[Project Page](mias.group/GCM) | [Paper](https://arxiv.org/abs/2403.05388) | [Video]() <br/>


## Environment Setup
This repository is created using Anaconda and requires Python 3.7+.
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install tqdm pillow numpy matplotlib scipy opencv pyyaml pandas
```


## Our trained model
For your convenience, we provide a distilled patch descriptor in the `models/trained` folder.


## Evaluation
### Prepare the data and off-the-shelf models
#### Prepare the data
Please follow the [IME](https://github.com/ufukefe/IME) using `hpatches_organizer.py` to organize your dataset.
```bash
cd Datasets

wget -P Datasets http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz

tar -xvzf hpatches-sequences-release.tar.gz &> /dev/null

# Remove the high-resolution sequences
cd hpatches-sequences-release
rm -rf i_contruction i_crownnight i_dc i_pencils i_whitebuilding v_artisans v_astronautis v_talent
cd ..

python hpatches_organizer.py

rm *.tar.gz
rm -rf hpatches-sequences-release
rm -rf hpatches_organizer.py
```

#### Prepare off-the-shelf models (optional)
The off-the-shelf models used are listed below. **Note that these models are not required** if you only want to use the distilled patch descriptor along with the image classification or semantic segmentation models.  
- [R2D2](https://github.com/naver/r2d2/tree/master/models)
- [DeepPruner](https://github.com/uber-research/DeepPruner/tree/master/deeppruner#Weights)  
- [PSMNet](https://github.com/JiaRenChang/PSMNet?tab=readme-ov-file#pretrained-model)  
- [CREStereo-Pytorch](https://github.com/ibaiGorordo/CREStereo-Pytorch)

The PSMNet and CREStereo models have been simplified since we only use backbone.
The relevant weights files can be downloaded at [here](https://pan.baidu.com/s/1Y8EjijDwh2LRsZIRocEMRQ) (extract code 0208). 
Note that [CREStereo-Pytorch](https://github.com/ibaiGorordo/CREStereo-Pytorch) are non-official Pytorch implementations, official repository: [CREStereo](https://github.com/megvii-research/CREStereo).

Place the data and off-the-shelf models as below:
```
GCM
├── configs
├── Datasets
│   ├── hpatches
│   ├── ~~hpatches-sequences-release~~
│   ├── eval_hpatches.py
│   └── ~~hpatches_organizer.py~~
├── models
│   ├── off-the-shelf
│   │   ├── CREStereo
│   │   |   └── backbone.pth
│   │   ├── DeepPruner
│   │   |   └── DeepPruner-fast-kitti.tar
│   │   ├── PSMNet
│   │   |   └── KITTI2012.pth
│   │   └── R2D2
│   │   |   └── r2d2_WASF_N16.pt
│   └── trained
│   │   └── dstl_r2d2.pt
├── nets
└── other files...
```


### Evaluation on HPatches
The evaluation is based on the code from [IME](https://github.com/ufukefe/IME).

#### Extract and process output
First, extract and save the original algorithm's output  
```bash
python algorithm_wrapper_util.py --config configs/VGG_dstlR2D2_95.yml --output_dir /path/to/output
```
Then, read saved outputs and transform to proper format (keypointsA, keypointsB, matches)  
```bash
python algorithm_wrapper.py --output_dir /path/to/output
```

#### Evaluate results
```bash
python Datasets/eval_hpatches.py --algorithms output1, output2, ...
```

#### Example
```bash
python algorithm_wrapper_util.py --config configs/VGG_dstlR2D2_95.yml --output_dir Results/hpatches/COARSEVGG19_dstlR2D2_95
python algorithm_wrapper.py --output_dir Results/hpatches/COARSEVGG19_dstlR2D2_95

python algorithm_wrapper_util.py --config configs/VGG_dstlR2D2_6.yml --output_dir Results/hpatches/COARSEVGG19_dstlR2D2_6
python algorithm_wrapper.py --output_dir Results/hpatches/COARSEVGG19_dstlR2D2_6

python Datasets/eval_hpatches.py --algorithms COARSEVGG19_dstlR2D2_95, COARSEVGG19_dstlR2D2_6
```

## Supported models
- VGG19
- VGG19_BN
- RESNET18
- RESNET50
- DEEPLABV3_RESNET50
- DEEPLABV3_MOBILE
- FCN_RESNET50
- DEEPPRUNER_FATS
- PSM
- CRE
- COARSEVGG19
- R2D2
- SWIN_TRANSFORMER
- RESNEXT50
- EFFICIENTNETV2
- MOBILEV3

You can use different backbone using the text listed above in the yaml file.


## Train
We provide our network and loss function in `utils/train` folder.
If you want to train a distilled patch descriptor, please clone the [R2D2 repository](https://github.com/naver/r2d2).
Please replace `patchnet.py`, then add `dstltrain.py` and `distillation_feature_loss.py`.

# Acknowledgements
Part of the code is from previous works:
- [R2D2](https://github.com/naver/r2d2)  
- [IME](https://github.com/ufukefe/IME)  
- [DFM](https://github.com/ufukefe/DFM)  
- [CREStereo-Pytorch](https://github.com/ibaiGorordo/CREStereo-Pytorch)
- [DeepPruner](https://github.com/uber-research/DeepPruner)  
- [PSMNet](https://github.com/JiaRenChang/PSMNet)  

We thank all authors for open-sourcing their projects. 
