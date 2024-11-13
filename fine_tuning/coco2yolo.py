from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="prosegur_pet-person_clean-merged_coco",
    save_dir="prosegur_pet-person_clean-merged_yolo/",
    use_segments=False,
    use_keypoints=False,
    cls91to80=True,
)
