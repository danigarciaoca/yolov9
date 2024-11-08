import cv2
import numpy as np
import torch

from .src import ROOT
from .src.models.common import DetectMultiBackend
from .src.utils.augmentations import letterbox
from .src.utils.general import check_img_size, non_max_suppression, scale_boxes
from .src.utils.plots import colors
from .src.utils.torch_utils import select_device, smart_inference_mode

torch.backends.cudnn.benchmark = True  # faster for fixed-size inference


def box_label(im, box, label='', color=(128, 128, 128), lw=2, txt_color=(255, 255, 255)):
    # Add one xyxy box to image with label
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)


class YOLOv9:
    @smart_inference_mode()
    def __init__(self,
                 weights='yolo.pt',  # model path or triton URL
                 data=ROOT / 'data/coco.yaml',  # dataset.yaml path
                 imgsz=(640, 640),  # inference size (height, width)
                 device="cuda:0",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 half=False,  # use FP16 half-precision inference
                 dnn=False,  # use OpenCV DNN for ONNX inference
                 ):
        """ Initializes the YOLOv9 model for object detection. """
        device = select_device(device)
        self.half = half

        # load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = model.stride, model.names, model.pt
        self.device, self.fp16 = model.device, model.fp16

        # warm up model
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup

        self.model = model

    @smart_inference_mode()
    def __call__(self,
                 img,  # image to process
                 classes=None,  # filter by class: --class 0, or --class 0 2 3
                 conf_thres=0.25,  # confidence threshold
                 iou_thres=0.45,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 agnostic_nms=False,  # class-agnostic NMS
                 augment=False,  # augmented inference
                 visualize=False  # visualize features
                 ):
        """ Runs object detection on the input image. """
        # input image pre-processing
        img_sz = img.shape[:2]  # input image size (before pre-processing)
        img = self.preprocess(img)
        img_sz_p = img.shape[2:]  # input image size (after pre-processing)

        # inference
        pred = self.model(img, augment=augment, visualize=visualize)

        # NMS post-processing
        pred = self.postprocess(pred, img_sz_p, img_sz, classes, conf_thres, iou_thres, max_det, agnostic_nms)

        return pred

    def preprocess(self, image):
        """ Prepares an image for YOLOv9 model input. """
        # Pad and resize
        im = letterbox(image, self.imgsz, stride=self.stride, auto=self.pt)[0]  # padded resize

        # Reorder dimensions change channel order (color space conversion)
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        # Cast and normalize
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0

        # Add batch dimension
        im = im[None]  # expand for batch dim

        return im

    @staticmethod
    def postprocess(pred,
                    img_sz_p,
                    img_sz,
                    classes=None,  # filter by class: --class 0, or --class 0 2 3
                    conf_thres=0.25,  # confidence threshold
                    iou_thres=0.45,  # NMS IOU threshold
                    max_det=1000,  # maximum detections per image
                    agnostic_nms=False  # class-agnostic NMS
                    ):
        """ Processes YOLOv9 detection results, so they can be directly used. """
        # Filter unwanted detections
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Rescale boxes to original image size
        for det in pred:  # per image (batch cases)
            if len(det):
                det[:, :4] = scale_boxes(img_sz_p, det[:, :4], img_sz).round()

        return pred

    def annotate(self,
                 img,  # image to annotate
                 det,  # detections used for annotation
                 line_width=2,  # box line width
                 hide_labels=False,  # hide labels
                 hide_conf=False  # hide confidences
                 ):
        for *xyxy, conf, cls in reversed(det):
            # Draw bbox into the image
            c = int(cls)  # integer class
            label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
            box_label(img, xyxy, label, color=colors(c, True), lw=line_width)

        return img
