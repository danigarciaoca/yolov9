# How to run?

## Create conda environment

```bash
git clone https://github.com/danigarciaoca/yolov9.git
cd yolov9

conda env create -f environment.yml
conda activate yolov9
```

## Train

```bash
cd yolov9/src

python train_dual.py --batch 16 --epochs 25 --img 640 --device 0 --min-items 0 --close-mosaic 15 --data {dataset.location}/data.yaml --weights {weights.location}/yolov9-e.pt --cfg models/detect/yolov9-e.yaml --hyp data/hyps/hyp.scratch-high.yaml
```

## Export

```bash
python torch2trt.py
```

## Evaluate

```bash
cd evaluation

# generate predicted labels for test set
python 1_get_labels.py

# get AP (per class) and mAP
python 2_evaluate_model.py

# get FPS performance metric
python 3_measure_inference_time.py
```