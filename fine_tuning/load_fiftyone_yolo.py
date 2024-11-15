import fiftyone as fo

# dataset loading parameters
dataset_name = "prosegur_person_pet_balanced_2"  # FiftyOne dataset name
yaml_path = "prosegur_person_pet_balanced_2/data.yml"  # path to the YOLOv5-format dataset YAML file
persist = False  # make the dataset persistent

fo.delete_dataset(dataset_name)

if dataset_name in fo.list_datasets():
    # load dataset from database
    dataset = fo.load_dataset(dataset_name)
    print(f"Dataset '{dataset_name}' loaded from database.")
else:
    # load dataset from disk
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.YOLOv5Dataset,
        label_type="detections",
        yaml_path=yaml_path,
        split="val",
        name=dataset_name,
        persistent=persist
    )
    print(f"Dataset '{dataset_name}' loaded from disk.")

if __name__ == "__main__":
    # launch FiftyOne with the loaded dataset
    session = fo.launch_app(dataset)
    session.wait()  # block execution until the script is stopped
