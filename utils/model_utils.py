# -*- coding:utf-8 -*-
"""
    @Description:
    @Author: Yu Han and ZiWei Long
    @Date: 2023/06/01 20:55
    @Company: 
"""
import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from collections import namedtuple, OrderedDict
import matplotlib.pyplot as plt
import nets.r2d2.patchnet as r2d2
from nets.vgg19 import Vgg19, FineVgg19, CoarseVgg19
from nets.resnet50 import Resnet50
from nets.r2d2.tools import common

import os


def get_all_models():
    func_map = {
        "VGG19": get_vgg19,
        "VGG19_BN": get_vgg19_bn,
        "RESNET18": get_resnet18,
        "RESNET50": get_resnet50,
        # "UNET_ENCODER": get_unet_encoder,
        # "UNET": get_unet,
        "DEEPLABV3_RESNET50": get_deeplabv3_resnet50backbone,
        "DEEPLABV3_MOBILE": get_deeplabv3_mobilebackbone,
        "FCN_RESNET50": get_fcn_resnet50,
        "DEEPPRUNER_FATS": get_deeppruner_feature_extraction,
        "PSM": get_psmnet_feature_extraction,
        # "BG": get_bgnet_feature_extraction,
        "CRE": get_cre_stereo_feature_extraction,
        "COARSEVGG19": get_coarse_vgg19,
        # "QUANTIZED_RESNET18": get_quantized_resnet18,
        "R2D2": get_only_r2d2,
        "R2D2_STU": get_only_r2d2_student,
        "SWIN_TRANSFORMER": get_swin_transformer,
        "RESNEXT50": get_resnext50,
        "EFFICIENTNETV2": get_EfficientNetV2,
        "MOBILEV3": get_MobileNet_V3_Large,
    }
    return func_map


def get_all_fine_models():
    func_map = {
        "VGG19": get_fine_vgg19,
        "R2D2": get_r2d2,
        # "SUPERPOINT": get_SuperPoint
    }
    return func_map


def get_vgg19(ratio_th, device, **kwargs):
    assert len(ratio_th) == 6, 'ratio_th must be a list of 6 elements'
    print('loading VGG19...')
    model = Vgg19(batch_normalization=False).to(device).eval()
    print('model is loaded.')
    return model


def get_vgg19_bn(ratio_th, device, **kwargs):
    assert len(ratio_th) == 6, 'ratio_th must be a list of 6 elements'
    print('loading VGG19_BN...')
    model = Vgg19(batch_normalization=True).to(device).eval()
    print('model is loaded.')
    return model


def get_coarse_vgg19(ratio_th, device, **kwargs):
    print('loading CoarseVgg19...')
    model = CoarseVgg19(batch_normalization=False).to(device).eval()
    print('CoarseVgg19 model is loaded.')
    return model


def get_resnet50(ratio_th, device, **kwargs):
    assert len(ratio_th) == 5, 'ratio_th must be a list of 5 elements'
    print('loading ResNet50...')
    model = Resnet50().to(device).eval()
    print('ResNet50 model is loaded.')
    return model


def get_resnet18(ratio_th, device, down_sample=16, **kwargs):
    import warnings
    print('###' * 30)
    print('loading ResNet18...')
    result_dict = {'relu': 'f_map_dw2', 'layer1': 'f_map_dw4', 'layer2': 'f_map_dw8', 'layer3': 'f_map_dw16'}
    if down_sample == 8:
        result_dict = {'relu': 'f_map_dw2', 'layer1': 'f_map_dw2', 'layer4': 'f_map_dw8'}
        message = "Currently downsampling 8x, please note the ratio test list and the one-stage upsampling"
        warnings.warn(message, RuntimeWarning)

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights).to(device).eval()
    nb_of_weights = common.model_size(model)
    print(f" ( Model size: {nb_of_weights / 1000:.0f}K parameters )")

    new_m = models._utils.IntermediateLayerGetter(model, result_dict)
    print('ResNet18 model is loaded.')
    print('###' * 30)

    return new_m


def get_deeplabv3_resnet50backbone(ratio_th, device, **kwargs):
    print('loading deeplabv3 resnet50 backbone...')
    weights = models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = models.segmentation.deeplabv3_resnet50(weights=weights).backbone.to(device).eval()

    result_dict = {'conv1': 'feat1', 'layer1': 'feat2', 'layer4': 'feat3'}

    new_m = models._utils.IntermediateLayerGetter(model, result_dict)
    print('Deeplabv3 resnet50 backbone model is loaded.')
    return new_m


def get_deeplabv3_mobilebackbone(ratio_th, device, **kwargs):
    if "required_layers" in kwargs.keys():
        required_layers = kwargs["required_layers"]
    else:
        # 1 3 6 10 15
        required_layers = [1, 3, 6, 10, 15]

    assert len(ratio_th) == len(required_layers), 'ratio_th must be the same length as required_layers'
    print('loading deeplabv3 mobile backbone...')

    weights = models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
    model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights).backbone.eval()

    result_dict = {f"{i}": "feat" + str(i) for i in required_layers}

    new_m = models._utils.IntermediateLayerGetter(model, result_dict)
    new_m = new_m.to(device).eval()
    print('Deeplabv3 mobile backbone model is loaded.')
    return new_m


def get_fcn_resnet50(ratio_th, device, **kwargs):
    # assert len(ratio_th) == 3, 'ratio_th must be a list of 3 elements'

    weights = models.segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = models.segmentation.fcn_resnet50(weights=weights).backbone.eval()
    result_dict = {'conv1': 'feat1', 'layer1': 'feat2', 'layer4': 'feat5'}
    new_m = models._utils.IntermediateLayerGetter(model, result_dict)
    return new_m


def get_deeppruner_feature_extraction(ratio_th, device, **kwargs):
    # assert len(ratio_th) == 4, 'ratio_th must be a list of 4 elements'
    from nets.deeppruner.models.deeppruner import DeepPruner

    print('loading deeppruner feature extraction...')
    model = DeepPruner()
    model.eval()
    load_path = r"..\models\off-the-shelf\DeepPruner\DeepPruner-fast-kitti.tar"
    state_dict = torch.load(load_path)

    model.load_state_dict(state_dict['state_dict'], strict=False)
    fe = model.feature_extraction
    fe = fe.to(device)
    fe.eval()
    print('deeppruner feature extraction model is loaded.')

    return fe


def get_psmnet_feature_extraction(ratio_th, device, **kwargs):
    from nets.PSMNet.models.feature import PSMNet
    assert len(ratio_th) == 4, 'ratio_th must be a list of 4 elements'
    print('loading psmnet feature extraction...')
    model = PSMNet(192)
    load_path = r"models\off-the-shelf\PSMNet\KITTI2012.pth"
    state_dict = torch.load(load_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print('psmnet feature extraction model is loaded.')
    return model


# def get_bgnet_feature_extraction(ratio_th, device, **kwargs):
#     from nets.BGNet.models.bgnet import BGNet
#     assert len(ratio_th) == 5, 'ratio_th must be a list of 5 elements'
#     print('loading BGNet feature extraction...')
#     model = BGNet()
#     load_path = r"models\off-the-shelf\BGNet\kitti_12_BGNet.pth"
#     state_dict = torch.load(load_path)
#     model.load_state_dict(state_dict)
#     model.eval()
#     model.to(device)
#     print('BGNet feature extraction model is loaded.')
#     return model


def get_cre_stereo_feature_extraction(ratio_th, device, **kwargs):
    from nets.CREStereo.nets.crestereo import CREStereo

    print('loading CREStereo feature extraction...')
    model = CREStereo(max_disp=256, downsample=True, test_mode=True)
    load_path = r"models\off-the-shelf\CREStereo\feat.pth"
    state_dict = torch.load(load_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print('CREStereo feature extraction model is loaded.')
    return model


def get_quantized_resnet18(ratio_th, device, **kwargs):
    from torchvision.models.quantization import resnet18, ResNet18_QuantizedWeights
    weights = ResNet18_QuantizedWeights.DEFAULT
    model = resnet18(weights=weights, quantize=True).to(device).eval()

    # result_dict = {'relu': 'feat_dw2', 'layer1': 'feat_dw4', 'layer2': 'feat_dw8', 'layer3': 'feat_dw16'}
    # new_m = models._utils.IntermediateLayerGetter(model, result_dict).to(device).eval()

    return model


def get_only_r2d2(ratio_th, device, **kwargs):
    # path = r"models\off-the-shelf\R2D2\r2d2_WASF_N16.pt"
    path = r"models\off-the-shelf\R2D2\dstl_r2d2.pt"
    checkpoint = torch.load(path)
    # model = r2d2.Coarse_Quad_L2Net_ConfCFS()
    model = r2d2.Coarse_Student_Quad_L2Net_ConfCFS()
    print("\n>> loading net = " + "Coarse_Quad_L2Net_ConfCFS")
    nb_of_weights = common.model_size(model)
    print(f" ( Model size: {nb_of_weights / 1000:.0f}K parameters )")

    weights = checkpoint['state_dict']
    model.load_state_dict({k.replace('module.', ''): v for k, v in weights.items()})
    model.to(device).eval()
    return model


def get_only_r2d2_student(ratio_th, device, **kwargs):
    path = r"..\models\trained\dstl_r2d2.pt"
    checkpoint = torch.load(path)
    model = r2d2.Coarse_Student_Quad_L2Net_ConfCFS()
    print("\n>> loading net = " + "Student_Quad_L2Net_ConfCFS")
    nb_of_weights = common.model_size(model)
    print(f" ( Model size: {nb_of_weights / 1000:.0f}K parameters )")

    weights = checkpoint['state_dict']
    model.load_state_dict({k.replace('module.', ''): v for k, v in weights.items()})
    model.to(device).eval()
    return model


def get_swin_transformer(ratio_th, device, **kwargs):
    from nets.vit import ViT
    model = ViT()
    model.to(device).eval()
    return model


def get_resnext50(ratio_th, device, **kwargs):
    from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
    weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V2
    model = resnext50_32x4d(weights=weights).to(device).eval()

    nb_of_weights = common.model_size(model)
    print(f" ( Model size: {nb_of_weights / 1000:.0f}K parameters )")
    result_dict = {'relu': 'feat1', 'layer1': 'feat2', 'layer2': 'feat3', 'layer3': 'feat4'}

    new_m = models._utils.IntermediateLayerGetter(model, result_dict)
    print('ResNeXt50 model is loaded.')

    return new_m


def get_EfficientNetV2(ratio_th, device, **kwargs):
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = efficientnet_v2_s(weights=weights).features[:6].to(device).eval()

    nb_of_weights = common.model_size(model)
    print(f" ( Model size: {nb_of_weights / 1000:.0f}K parameters )")
    result_dict = {'1': 'feat1', '2': 'feat2', '3': 'feat3', '4': 'feat4', '5': 'feat4_copy'}
    model = models._utils.IntermediateLayerGetter(model, result_dict)
    model.to(device).eval()
    print('EfficientNetV2 model is loaded.')

    return model


def get_MobileNet_V3_Large(ratio_th, device, **kwargs):
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model = mobilenet_v3_large(weights=weights).features[:13].to(device).eval()

    result_dict = {'1': 'feat1', '3': 'feat2', '6': 'feat3', '11': 'feat4', '12': 'feat4_copy'}
    model = models._utils.IntermediateLayerGetter(model, result_dict)
    nb_of_weights = common.model_size(model)
    print(f" ( Model size: {nb_of_weights / 1000:.0f}K parameters )")
    model.to(device).eval()
    print('MobileNet V3 Large model is loaded.')
    return model


#######################################

def get_fine_vgg19(device, **kwargs):
    print('loading VGG19...')
    model = FineVgg19(batch_normalization=False).to(device).eval()
    print('model is loaded.')
    return model


def get_r2d2(device, **kwargs):
    # path = r"models\off-the-shelf\R2D2\r2d2_WASF_N8_big.pt"
    if "weights" in kwargs and kwargs["weights"] is not None:
        path = kwargs["weights"]
    else:
        path = r"..\models\off-the-shelf\R2D2\r2d2_WASF_N16.pt"
    print('###' * 30)
    print(f"path: {path}")
    model = r2d2.load_network(path).to(device).eval()
    print('###' * 30)
    return model


def get_SuperPoint(device, **kwargs):
    from nets.superpoint import SuperPoint
    print("load SuperPoint")
    model = SuperPoint(max_num_keypoints=2048).eval().to(device)
    print('###' * 30)
    return model


# Convert a dict to a namedtuple
def convert2namedtuple(outputs):
    if isinstance(outputs, OrderedDict):
        Outputs = namedtuple("Outputs", list(outputs.keys()))
        outputs = Outputs._make(list(outputs.values()))
    if len(outputs) != 1:
        if outputs[-1].shape[2:] != outputs[-2].shape[2:]:
            OutputsAddCopy = namedtuple("OutputsAddCopy", outputs._fields + ("copy",))
            _ = tuple(outputs) + (outputs[-1],)
            outputs = OutputsAddCopy(*_)
    return outputs
