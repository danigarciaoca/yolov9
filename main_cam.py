import time

import cv2

from yolov9 import YOLOv9


def main(det_model, video_source):
    # If no video source is provided, use the default webcam
    cap = cv2.VideoCapture(video_source)

    # Check if the video capture has been successfully initialized
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame capture failed, break the loop
        if not ret:
            print("Error: Could not read frame.")
            break

        # run object detection
        start = time.time()
        det = det_model(frame, classes=None, conf_thres=0.25)[0]  # keep item 0 (single image inference)
        fps = 1 / (time.time() - start)
        print(f"FPS: {fps}")

        # annotate frame with detections
        frame_out = model.annotate(frame, det, line_width=3)

        # Display the resulting frame
        cv2.imshow('Webcam', frame_out)

        # Exit if ESC key is pressed
        if cv2.waitKey(1) == 27:  # 27 is the ASCII code for ESC
            print("ESC key pressed. Exiting...")
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


# Run the main function
if __name__ == "__main__":
    # load model
    weights_path = "weights/yolov9-e-converted.pt"  # model weights
    model = YOLOv9(weights_path, half=False)

    # run inference on webcam stream
    main(det_model=model, video_source=0)

    print("Done!")
