import fiftyone as fo

# dataset loading parameters
dataset_name = "prosegur_person-pet_clean-merged"  # FiftyOne dataset name
data_path = "coco/test/images"  # directory containing the source images
labels_path = "coco/test/instances.json"  # path to the COCO labels JSON file
persist = False  # make the dataset persistent

# fo.delete_dataset(dataset_name)

if dataset_name in fo.list_datasets():
    # load dataset from database
    dataset = fo.load_dataset(dataset_name)
    print(f"Dataset '{dataset_name}' loaded from database.")
else:
    # load dataset from disk
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=labels_path,
        name=dataset_name,
        persistent=persist
    )
    print(f"Dataset '{dataset_name}' loaded from disk.")

if __name__ == "__main__":
    # launch FiftyOne with the loaded dataset
    session = fo.launch_app(dataset)
    session.wait()  # block execution until the script is stopped
