

OBJECT_DETECTION_BASE_PATH = '/media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/git_project/mot/'

DETR_CONFIG = OBJECT_DETECTION_BASE_PATH + "configs/object_detection/detr_256_6_6_torchvision.yaml"
DETR_WEIGHT = OBJECT_DETECTION_BASE_PATH + 'pretrained_models/object_detection/converted_model.pth'

DYHEAD_FPN_CONFIG = OBJECT_DETECTION_BASE_PATH + "configs/object_detection/dyhead_r50_atss_fpn_1x.yaml"
DYHEAD_FPN_WEIGHT = OBJECT_DETECTION_BASE_PATH + 'pretrained_models/object_detection/dyhead_r50_atss_fpn_1x.pth'

DYHEAD_SWINT_CONFIG = OBJECT_DETECTION_BASE_PATH + "configs/object_detection/dyhead_swint_atss_fpn_2x_ms.yaml"
DYHEAD_SWINT_WEIGHT = OBJECT_DETECTION_BASE_PATH + 'pretrained_models/object_detection/dyhead_swint_atss_fpn_2x_ms.pth'

def get_obj_det_config(arch_type):
    if arch_type == "detr":
        return DETR_CONFIG
    elif arch_type == 'dyhead_fpn':
        return DYHEAD_FPN_CONFIG
    elif arch_type == 'dyhead_swint':
        return DYHEAD_SWINT_CONFIG

def get_obj_det_weight(arch_type):
    if arch_type == "detr":
        return DETR_WEIGHT
    elif arch_type == 'dyhead_fpn':
        return DYHEAD_FPN_WEIGHT
    elif arch_type == 'dyhead_swint':
        return DYHEAD_SWINT_WEIGHT
