
DATASET_TRAIN = 'cityscapes_fine_instance_seg_train'
DATASET_VAL = 'cityscapes_fine_instance_seg_val'
MODEL_WEIGHT_PATH = 'out_model/object_det_models/dyhead_fpn/model_final.pth'
TARGET_DEVICE = 'cuda'

CITYSCAPES_CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
OBJ_DET_MODEL_OUTDIR = 'out_model/object_det_models'
VIDEO_OUTPUT_PATH='video_output'

# Use input/video/nvidia.mp4, cityscapes_long.mp4, pexels4_.mp4, mot17_1.mp4, car_video.mp4
VIDEO_INPUT_FILE_NAME = "input/video/pexels_3.mp4"
VIDEO_OUTPUT_FILE_NAME = 'video_output/kalman_result.mp4'

FONT_SCALE = 2e-3
THICKNESS_SCALE = 1e-3

DEBUG_FLAG = True

FRAME_WIDTH = 512
FRAME_HEIGHT = 512

