import numpy as np
import cv2
import time
import config
from netutils import ObjectDetectionArch, ObjDetArchType
from detectron2.structures import Boxes, RotatedBoxes
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

import yolov5
import tqdm
import math

class ObjectDetectorAPI:
    def __init__(self, arch_type):
        self.arch_type = arch_type
        self.target_device = config.TARGET_DEVICE

        self.object_detection = ObjectDetectionArch(len(config.CITYSCAPES_CLASSES), arch_type)
        self.object_detection.register_dataset(config.DATASET_TRAIN, config.DATASET_VAL)
        self.object_detection.set_model_weights(config.MODEL_WEIGHT_PATH)
        self.object_detection.set_target_device(self.target_device)
        self.object_detection.set_score_threshold(0.7)
        self.object_detection.set_confidence_threhold(0.7)

        self.object_detection.print_cfg()

        self.predictor = self.object_detection.default_predictor()


    def create_text_labels(self, classes, scores, class_names, is_crowd=None):
        """
        Args:
            classes (list[int] or None):
            scores (list[float] or None):
            class_names (list[str] or None):
            is_crowd (list[bool] or None):

        Returns:
            list[str] or None
        """
        labels = None
        if classes is not None:
            if class_names is not None and len(class_names) > 0:
                labels = [class_names[i] for i in classes]
            else:
                labels = [str(i) for i in classes]
        if scores is not None:
            if labels is None:
                labels = ["{:.0f}%".format(s * 100) for s in scores]
            else:
                labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
        if labels is not None and is_crowd is not None:
            labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
        return labels

    def convert_boxes(self,
                      boxes):
        """
        Convert different format of boxes to an NxB array, where B = 4 or 5 is the box dimension.
        """
        if isinstance(boxes, Boxes) or isinstance(boxes, RotatedBoxes):
            return boxes.tensor.detach().numpy()
        else:
            return np.asarray(boxes)

    def filter_predictions_from_outputs(self,
                                        outputs,
                                        threshold=0.5,
                                        verbose=True):
        predictions = outputs["instances"].to("cpu")
        #print("predictions: {}".format(predictions))
        if verbose:
            print(list(predictions.get_fields()))

        # Reference: https://github.com/facebookresearch/detectron2/blob/7f06f5383421b847d299b8edf480a71e2af66e63/detectron2/structures/instances.py#L27
        #
        #   Indexing: ``instances[indices]`` will apply the indexing on all the fields
        #   and returns a new :class:`Instances`.
        #   Typically, ``indices`` is a integer vector of indices,
        #   or a binary mask of length ``num_instances``

        indices = [i
                   for (i, s) in enumerate(predictions.scores)
                   if s >= threshold
                   ]
        print("indices: {}".format(indices))
        filtered_predictions = predictions[indices]

        return filtered_predictions

    def processFrame(self, image):
        print("processFrame shape of image{}".format(image.shape))
        print("====PREDICTION======= STARTS")
        start = time.time()
        outputs = self.predictor(image)
        end = time.time()
        elapsed_time = (end - start) * 1000
        print("Evaluation Time for arch: {} on device: {} is {} ms ".format(self.arch_type, self.target_device, elapsed_time))
        filtered_outputs = self.filter_predictions_from_outputs(outputs)
        pred_boxes = filtered_outputs.pred_boxes
        #print("Length of pred_boxes: {}, pred_boxes: {}".format(len(pred_boxes), pred_boxes))
        boxes_list = self.convert_boxes(pred_boxes)
        #print("Length of boxes: {}, boxes: {}".format(len(boxes_list), boxes_list))

        scores = filtered_outputs.scores
        #print("Length of scores: {}, boxes: {}".format(len(scores), scores))

        pred_classes = filtered_outputs.pred_classes
        #print("Length of pred_classes: {}, pred_classes: {}".format(len(pred_classes), pred_classes))

        self.num_detections = len(boxes_list)

        return boxes_list, scores.tolist(), pred_classes.tolist(), self.num_detections

    def close(self):
        pass
    def get_predictor(self):
        return self.predictor
    def get_cfg(self):
        return self.object_detection.get_net_cfg()

def runOnVideo(cap, maxFrames, odapi):
    readFrames = 0
    # Initialize visualizer
    v = VideoVisualizer(MetadataCatalog.get(odapi.get_cfg().DATASETS.TRAIN[0]), ColorMode.IMAGE)
    while True:
        r, img = cap.read()
        predictor = odapi.get_predictor()
        outputs = predictor(img)
        filtered_output = odapi.filter_predictions_from_outputs(outputs)
        # Make sure the frame is colored
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Draw a visualization of the predictions using the video visualizer
        #visualization = v.draw_instance_predictions(frame, filtered_output["instances"].to("cpu"))
        visualization = v.draw_instance_predictions(frame, filtered_output)
        # Convert Matplotlib RGB format to OpenCV BGR format
        visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)
        yield visualization

        readFrames += 1
        if readFrames > maxFrames:
            break
def draw_bboxes(img, bbox, identities=None, offset=(0,0)):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = config.COLORS_150[id%len(config.COLORS_150)]
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    return

def runOnVideoWith(cap, maxFrames):
    threshold = 0.5
    readFrames = 0
    #img = cv2.resize(img, (1280, 720))
    while True:
        r, img = cap.read()
        boxes, scores, classes, num = odapi.processFrame(img)
        print("boxes: {}".format(boxes))
        print("classes: {}".format(classes))
        cityscapes_metadata = MetadataCatalog.get("cityscapes_fine_instance_seg_train")

        labels = odapi.create_text_labels(classes, scores, cityscapes_metadata.get("thing_classes", None))
        print("labels: {}".format(labels))

        #draw_bboxes(img, boxes, identities=labels)
        FONT_SCALE = 2e-3
        THICKNESS_SCALE = 1e-3

        # Visualization of the results of a detection.
        #for i in range(len(boxes)):
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(i) for i in box]
            width = x2-x1
            height = y2-y1
            # Class 2 represents Car
            if scores[i] > threshold:
                font_scale = min(width, height) * FONT_SCALE
                wide = math.ceil(min(width, height) * THICKNESS_SCALE)
                #box = boxes[i]
                #cv2.rectangle(img,(int(box[1]),int(box[0])),(int(box[3]),int(box[2])),(255,0,0),2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                t_size = cv2.getTextSize(labels[i], cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
                cv2.rectangle(img, (x1, y1-2), (x1 + t_size[0] + 3, (y1-20) + t_size[1] + 4), (255, 255, 0), -1)
                cv2.putText(img, labels[i], (x1, (y1-15) + t_size[1] + 4), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                               fontScale=font_scale, thickness=wide, color=[255,255,255])

        image = img.copy()

        yield image
        readFrames += 1
        if readFrames > maxFrames:
            break

if __name__ == "__main__":
    odapi = ObjectDetectorAPI(ObjDetArchType.DYHEAD_FPN)
    threshold = 0.5
    cap = cv2.VideoCapture(config.VIDEO_INPUT_FILE_NAME)

    output_fname = 'video_output/obj_detection_api_result.mp4'
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    video_writer = cv2.VideoWriter(output_fname, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second),
                                   frameSize=(width, height), isColor=True)

    # Create a cut-off for debugging
    num_frames = 300

    # Enumerate the frames of the video
    for visualization in tqdm.tqdm(runOnVideoWith(cap, num_frames), total=num_frames):
        # Write to video file
        video_writer.write(visualization)

    # Release resources
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
