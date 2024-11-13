import time
from pathlib import Path

import cv2
from tqdm import tqdm

from yolov9 import YOLOv9


def process_image(det_model, img_path):
    # load input image
    img = cv2.imread(img_path)

    # run object detection ([0, 15, 16] -> {0: 'person', 15: 'cat', 16: 'dog'})
    start = time.time()
    _ = det_model(img, classes=[0, 15, 16], conf_thres=0.25, max_det=1000)[0]  # single image, so just keep item 0
    inference_time = time.time() - start

    return inference_time


if __name__ == "__main__":
    # input images folder
    images_dir = Path("coco/test/images")  # folder containing input images to be processed

    # load model
    weights_path = "../weights/yolov9-e-converted.engine"  # model weights
    model = YOLOv9(weights_path, half=True)

    # run inference and save inference times
    time_measures = []
    for img_path in tqdm(images_dir.iterdir(), total=2279):
        infer_t = process_image(det_model=model, img_path=img_path)
        time_measures.append(infer_t)

    # get some basic inference time statistics (min, max and mean)
    min_t, max_t, mean_t = min(time_measures), max(time_measures), sum(time_measures) / len(time_measures)
    min_fps, max_fps, mean_fps = 1 / max_t, 1 / min_t, 1 / mean_t
    print(f"min FPS: {min_fps} | mean FPS: {mean_fps} | max FPS: {max_fps}")

    print("Done!")
