from yolov9 import run

# Global options
weights = "weights/yolov9-e-converted.pt"  # weights path
imgsz = (640, 640)  # image (height, width)
batch_size = 1  # batch size
device = "cuda:0"  # cuda device, i.e. 0 or 0,1,2,3 or cpu
include = ["engine"]  # include formats
inplace = False  # set YOLO Detect() inplace=True

# ONNX options
opset = 12  # ONNX: opset version
dynamic = False  # ONNX/TF/TensorRT: dynamic axes
simplify = False  # ONNX: simplify model

# TensorRT options
half = False  # FP16 half-precision export
workspace = 4  # TensorRT: workspace size (GB)
verbose = False  # TensorRT: verbose log

# Export model
run(weights=weights,
    imgsz=imgsz,
    batch_size=batch_size,
    device=device,
    include=include,
    inplace=inplace,
    opset=opset,
    dynamic=dynamic,
    simplify=simplify,
    half=half,
    workspace=workspace,
    verbose=verbose
    )
