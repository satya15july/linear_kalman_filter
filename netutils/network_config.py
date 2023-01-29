import sys

CENTREMASK_LOC = '/media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/project/segmentation/projects_detectron2/instance_segmentation/centermask2'
sys.path.append(CENTREMASK_LOC)

sys.path.append('/media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/project/segmentation/AdelaiDet/adet/')

BASE_PATH = "/media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/git_project/segmentation/instance-segmentation/"
CENTREMASK_LITE_MV2_CONFIG = BASE_PATH + 'configs/centermask/centermask_lite_Mv2_FPN_ms_4x.yaml'
CENTREMASK_LITE_MV2__WEIGHT = BASE_PATH + 'pretrained_model/centermask/centermask_lite_Mv2_ms_4x.pth'

CENTREMASK_LITE_V19_SLIM_DW_CONFIG = BASE_PATH + 'configs/centermask/centermask_lite_V_19_slim_dw_eSE_FPN_ms_4x.yaml'
CENTREMASK_LITE_V19_SLIM_DW_WEIGHT = BASE_PATH + 'pretrained_model/centermask/centermask-lite-V-19-eSE-slim-dw-FPN-ms-4x.pth'

MASKRCNN_R50_CONFIG = '../COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
MASKRCNN_R50_WEIGHT = '../COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'

SOLOV2_R50_CONFIG = BASE_PATH + 'configs/solov2/R50_3x.yaml'
SOLOV2_R50_WEIGHT = BASE_PATH + 'pretrained_model/solov2/SOLOv2_R50_3x.pth'

BLENDMASK_DLA_CONFIG = BASE_PATH + 'blendmask/DLA_34_syncbn_4x.yaml'
BLENDMASK_DLA_WEIGHT = BASE_PATH + 'pretrained_model/blendmask/DLA_34_syncbn_4x.pth'

CONDINST_CONFIG = BASE_PATH + 'configs/condinst/MS_R_50_3x.yaml'
CONDINST_WEIGHT = BASE_PATH + 'pretrained_model/condinst/CondInst_MS_R_50_3x.pth'

TENSORMASK_CONFIG_6X = BASE_PATH + 'configs/tensormask/tensormask_R_50_FPN_6x.yaml'
TENSORMASK_WEIGHT_6X = BASE_PATH + 'pretrained_model/tensormask/model_final_e8df31_6x.pkl'


def get_net_config(arch_type):
    if arch_type == "condinst":
        return CONDINST_CONFIG
    elif arch_type == "solov2":
        return SOLOV2_R50_CONFIG
    elif arch_type == "centermask_mv2":
        return CENTREMASK_LITE_MV2_CONFIG
    elif arch_type == "centermask_v19_slim":
        return CENTREMASK_LITE_V19_SLIM_DW_CONFIG
    elif arch_type == 'tensormask':
        return TENSORMASK_CONFIG_6X


def get_net_pretrained_weight(arch_type):
    if arch_type == "condinst":
        return CONDINST_WEIGHT
    elif arch_type == "solov2":
        return SOLOV2_R50_WEIGHT
    elif arch_type == "centermask_mv2":
        return CENTREMASK_LITE_MV2__WEIGHT
    elif arch_type == "centermask_v19_slim":
        return CENTREMASK_LITE_V19_SLIM_DW_WEIGHT
    elif arch_type == 'tensormask':
        return TENSORMASK_WEIGHT_6X

