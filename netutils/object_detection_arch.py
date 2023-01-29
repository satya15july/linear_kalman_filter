import sys
sys.path.append('/media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/git_project/mot/ext_detectron2_net/detr/')
sys.path.append('/media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/git_project/mot/ext_detectron2_net/DynamicHead/')

from .object_detection_config import get_obj_det_config, get_obj_det_weight

from .predictor import VisualizationDemo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg

from .dyhead_trainer import DyHeadTrainer

from d2.train_net import Trainer
from d2.detr import add_detr_config

from dyhead import add_dyhead_config
from extra import add_extra_config

import cv2, os
import tqdm

from enum import Enum

class ObjDetArchType(Enum):
    DETR = 1
    DYHEAD_FPN = 2
    DYHEAD_SWINT = 3


class ObjectDetectionArch(object):
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
        if self.__arch_type == ObjDetArchType.DETR:
            self.cfg = get_cfg()
            add_detr_config(self.cfg)
            self.cfg.merge_from_file(get_obj_det_config('detr'))
            self.cfg.MODEL.WEIGHTS = get_obj_det_weight('detr')
            self.cfg.MODEL.DETR.NUM_CLASSES = self.num_classes
        elif self.__arch_type == ObjDetArchType.DYHEAD_FPN:
            self.cfg = get_cfg()
            add_dyhead_config(self.cfg)
            add_extra_config(self.cfg)
            self.cfg.merge_from_file(get_obj_det_config("dyhead_fpn"))
            self.cfg.MODEL.WEIGHTS = get_obj_det_weight('dyhead_fpn')
            self.cfg.MODEL.ATSS.NUM_CLASSES = self.num_classes
        elif self.__arch_type == ObjDetArchType.DYHEAD_SWINT:
            self.cfg = get_cfg()
            add_dyhead_config(self.cfg)
            add_extra_config(self.cfg)
            self.cfg.merge_from_file(get_obj_det_config('dyhead_swint'))
            self.cfg.MODEL.WEIGHTS = get_obj_det_weight('dyhead_swint')
            self.cfg.MODEL.ATSS.NUM_CLASSES = self.num_classes

        if self.__arch_type == ObjDetArchType.DYHEAD_SWINT:
            self.cfg.SOLVER.IMS_PER_BATCH = 1
        else:
            self.cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
        self.cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        self.cfg.SOLVER.MAX_ITER = 10000
        self.cfg.SOLVER.STEPS = []  # do not decay learning rate
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        #self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.OUTPUT_DIR = self.model_output

    def set_epochs(self, num):
        self.cfg.SOLVER.MAX_ITER = num

    def train(self, resume_flag=False):
        trainer = None
        if self.__arch_type == ObjDetArchType.DETR:
            trainer = Trainer(self.cfg)
        elif self.__arch_type == ObjDetArchType.DYHEAD_FPN or ObjDetArchType.DYHEAD_SWINT:
            trainer = DyHeadTrainer(self.cfg)

        trainer.resume_or_load(resume=resume_flag)
        trainer.train()

    def print_cfg(self):
        print("cfg: {}".format(self.cfg))

    def get_net_cfg(self):
        return self.cfg

    def set_score_threshold(self, value = 0.7):
        pass

    def set_confidence_threhold(self, value = 0.7):
        pass
        #self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = value

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
