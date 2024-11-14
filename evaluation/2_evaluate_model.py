import fiftyone as fo
import fiftyone.utils.yolo as fouy

# ground truth dataset
gt_dataset_name = "prosegur_gt"  # FiftyOne dataset name
data_path = "coco/test/images"  # directory containing the source images
gt_labels_path = "coco/test/instances.json"  # path to the COCO labels JSON file
persist = False  # make the dataset persistent

# predictions
# pred_labels_path = "coco_test_yolov9_coco/labels/test"  # path to the YOLOv5-format dataset predicted labels
pred_labels_path = "coco_test_yolov9_psg/labels/test"  # path to the YOLOv5-format dataset predicted labels

fo.delete_dataset(gt_dataset_name)

# Load COCO
if gt_dataset_name in fo.list_datasets():
    # load dataset from database
    dataset = fo.load_dataset(gt_dataset_name)
    print(f"Dataset '{gt_dataset_name}' loaded from database.")
else:
    # load dataset from disk
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=gt_labels_path,
        label_field="ground_truth",
        label_types="detections",
        name=gt_dataset_name,
        persistent=persist
    )
    print(f"Dataset '{gt_dataset_name}' loaded from disk.")

# add model predictions
classes = dataset.distinct("ground_truth.detections.label")
fouy.add_yolo_labels(
    dataset,
    "predictions",
    pred_labels_path,
    classes
)

if __name__ == "__main__":
    # # launch FiftyOne with the loaded dataset
    # session = fo.launch_app(dataset)
    # session.wait()  # block execution until the script is stopped

    # Performs an IoU sweep so that mAP and PR curves can be computed
    results = dataset.evaluate_detections(
        pred_field="predictions",
        gt_field="ground_truth",
        method="coco",
        compute_mAP=True,
    )

    # print mAP and AP results
    print(f"\nmAP: {results.mAP():.4f}")
    for cls in classes:
        ap_cls = results.mAP(classes=[cls])
        print(f" - class '{cls}' AP@0.5: {ap_cls:.4f}")

    print("Done!")
