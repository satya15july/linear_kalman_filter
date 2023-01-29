from .network_config import get_net_config, get_net_pretrained_weight
from .adet_trainer import AdetTrainer
from .centremask_trainer import CenterMaskTrainer
from .predictor import VisualizationDemo
from .tensormask_config import add_tensormask_config
from .tensormask_trainer import TensorMaskTrainer

from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2 import model_zoo

import cv2, os
import tqdm

from enum import Enum

class ArchType(Enum):
    MaskRCNN = 1
    CondInsta = 2
    SoloV2 = 3
    CentermaskLite_MV2 = 4
    CentermaskLite_V19_SLIM_DW = 5
    TensorMask = 6

class InstanceNetArch(object):
    def __init__(self, num_classes , archtype):
        self.num_classes = num_classes
        self.model_output = ''
        self.__arch_type = archtype
        self.set_net_config()

    def register_dataset(self, train_dataset, val_dataset):
        self.cfg.DATASETS.TRAIN = (train_dataset,)
        self.cfg.DATASETS.TEST = (val_dataset, )

    def set_model_output_path(self, model_output):
        self.model_output = model_output
        self.cfg.OUTPUT_DIR = self.model_output

    def set_target_device(self, target):
        self.cfg.MODEL.DEVICE = target

    def set_model_weights(self, model_weight):
        self.cfg.MODEL.WEIGHTS = model_weight

    def set_net_config(self):
        if self.__arch_type == ArchType.MaskRCNN:
            from detectron2.config import get_cfg
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif self.__arch_type == ArchType.CondInsta:
            from adet.config import get_cfg
            self.cfg = get_cfg()
            self.cfg.merge_from_file(get_net_config("condinst"))
            self.cfg.MODEL.WEIGHTS = get_net_pretrained_weight('condinst')
            self.cfg.MODEL.FCOS.NUM_CLASSES = self.num_classes
        elif self.__arch_type == ArchType.SoloV2:
            from adet.config import get_cfg
            self.cfg = get_cfg()
            self.cfg.merge_from_file(get_net_config('solov2'))
            self.cfg.MODEL.WEIGHTS = get_net_pretrained_weight('solov2')
            self.cfg.MODEL.FCOS.NUM_CLASSES = self.num_classes
        elif self.__arch_type == ArchType.CentermaskLite_MV2:
            from centermask2.centermask.config import get_cfg
            self.cfg = get_cfg()
            self.cfg.merge_from_file(get_net_config('centermask_mv2'))
            self.cfg.MODEL.WEIGHTS = get_net_pretrained_weight('centermask_mv2')
            self.cfg.MODEL.FCOS_CENTERMASK.NUM_CLASSES = self.num_classes
        elif self.__arch_type == ArchType.CentermaskLite_V19_SLIM_DW:
            from centermask2.centermask.config import get_cfg
            self.cfg = get_cfg()
            self.cfg.merge_from_file(get_net_config('centermask_v19_slim'))
            self.cfg.MODEL.WEIGHTS = get_net_pretrained_weight('centermask_v19_slim')
            self.cfg.MODEL.FCOS_CENTERMASK.NUM_CLASSES = self.num_classes
        elif self.__arch_type == ArchType.TensorMask:
            from detectron2.config import get_cfg
            self.cfg = get_cfg()
            add_tensormask_config(self.cfg)
            self.cfg.merge_from_file(get_net_config('tensormask'))
            self.cfg.MODEL.WEIGHTS = get_net_pretrained_weight('tensormask')
            self.cfg.MODEL.TENSOR_MASK.NUM_CLASSES = self.num_classes

        self.cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
        self.cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        self.cfg.SOLVER.MAX_ITER = 10000
        self.cfg.SOLVER.STEPS = []  # do not decay learning rate
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.OUTPUT_DIR = self.model_output

    def set_epochs(self, num):
        self.cfg.SOLVER.MAX_ITER = num

    def train(self, resume_flag=False):
        trainer = None
        if self.__arch_type == ArchType.MaskRCNN:
            trainer = DefaultTrainer(self.cfg)
        elif self.__arch_type == ArchType.CentermaskLite_MV2 or self.__arch_type == ArchType.CentermaskLite_V19_SLIM_DW:
            trainer = CenterMaskTrainer(self.cfg)
        elif self.__arch_type == ArchType.CondInsta or self.__arch_type == ArchType.SoloV2:
            trainer = AdetTrainer(self.cfg)
        elif self.__arch_type == ArchType.TensorMask:
            trainer = TensorMaskTrainer(self.cfg)

        trainer.resume_or_load(resume=resume_flag)
        trainer.train()

    def print_cfg(self):
        print("cfg: {}".format(self.cfg))

    def get_net_cfg(self):
        return self.cfg

    def set_score_threshold(self, value = 0.7):
        if self.__arch_type == ArchType.TensorMask:
            self.cfg.MODEL.TENSOR_MASK.SCORE_THRESH_TEST
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = value

    def set_confidence_threhold(self, value = 0.7):
        self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = value

    def default_predictor(self):
        return DefaultPredictor(self.cfg)

    def run_on_webcam(self):
        demo = VisualizationDemo(self.cfg)
        WINDOW_NAME = "arch = {}".format(self.__arch_type)
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()

    def run_on_video_input(self, video_input, output):
        demo = VisualizationDemo(self.cfg)
        video = cv2.VideoCapture(video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(video_input)

        if output:
            if os.path.isdir(output):
                output_fname = os.path.join(output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                #fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fourcc = cv2.VideoWriter_fourcc(*"mp4v"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if output:
            output_file.release()
        else:
            cv2.destroyAllWindows()

