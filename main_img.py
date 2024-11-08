import cv2

from yolov9 import YOLOv9


def main(det_model, image_source):
    # load input image
    img = cv2.imread(image_source)

    # run object detection
    det = det_model(img, classes=None, conf_thres=0.25, max_det=1000)[0]  # no batch inference, so just keep item 0

    # annotate image with detections
    img_out = model.annotate(img, det, line_width=3)

    # show annotated image
    cv2.imshow('Annotated image', img_out)

    # wait until a key is pressed, then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # load model
    weights_path = "weights/yolov9-e-converted.pt"  # model weights
    model = YOLOv9(weights_path, half=False)

    # run inference
    img_path = "img/04_2024-03-08_19-28-18-home_00041.jpg"  # input image
    main(det_model=model, image_source=img_path)

    print("Done!")
