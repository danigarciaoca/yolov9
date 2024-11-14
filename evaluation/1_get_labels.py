from pathlib import Path

import cv2
from tqdm import tqdm

from yolov9 import YOLOv9

# mapping from COCO class IDs to MPA class IDs
coco2mpa = {0: 2, 15: 0, 16: 1}


def process_image(det_model, in_img_path, out_labels_path, view_img):
    # load input image
    img = cv2.imread(in_img_path)

    # run object detection ([0, 15, 16] -> {0: 'person', 15: 'cat', 16: 'dog'})
    det = det_model(img, classes=[0, 15, 16], conf_thres=0.1, max_det=1000)[0]  # single image, so just keep item 0

    # annotate image with detections
    txt_out = out_labels_path / in_img_path.stem
    img_out = model.annotate(img, det, line_width=3, annot_img=view_img, save_txt=True,
                             save_conf=True, txt_file=txt_out, class_map=coco2mpa)

    if view_img:
        # show annotated image
        cv2.imshow('Annotated image', img_out)

        # wait until a key is pressed, then close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # input images folder
    images_dir = Path("coco/test/images")  # folder containing input images to be processed

    # output predicted labels path (class + bbox)
    labels_path = Path("coco_test_yolov9/labels/test")
    labels_path.mkdir(parents=True, exist_ok=True)

    # load model
    weights_path = "../weights/yolov9-e-converted.engine"  # model weights
    model = YOLOv9(weights_path, half=True)

    # run inference
    for img_path in tqdm(images_dir.iterdir(), total=2279):
        process_image(det_model=model, in_img_path=img_path, out_labels_path=labels_path, view_img=False)

    print("Done!")
