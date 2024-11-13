import random
import shutil
from pathlib import Path

# Set the seed
random.seed(42)


def _read_file_lines(path):
    with open(path, "r") as f:
        lines = [l.strip() for l in f.read().splitlines()]
        return [l for l in lines if l]


def _parse_yolo_row(row, classes):
    row_vals = row.split()

    cls_id = int(row_vals[0])

    label = classes[cls_id]

    return cls_id, label


def get_labels(annot_path, classes):
    labels = []
    cls_ids = []

    for row in _read_file_lines(annot_path):
        cls_id, label = _parse_yolo_row(row, classes)

        cls_ids.append(cls_id)
        labels.append(label)

    return cls_ids, labels


if __name__ == "__main__":
    # path to YOLO dataset
    split = "train"
    dataset_base_path = Path("prosegur_pet-person_clean-merged_yolo")
    annotations_path = dataset_base_path / "labels" / split

    classes = ["cat", "dog", "person"]

    # get those images with more than `num_people` people
    medium = 10
    difficult = 20
    extreme = 30
    cat_samples = []
    dog_samples = []
    many_people_samples = {"medium": [], "difficult": [], "extreme": []}
    for annot_file in annotations_path.iterdir():
        # get labels from the annotation file of current sample
        cls_ids, labels = get_labels(annot_file, classes)

        # count the occurrences of each class in current sample
        cat_count = labels.count("cat")
        dog_count = labels.count("dog")
        people_count = labels.count("person")

        # keep only-cat samples (no people)
        cat_samples.append(str(annot_file)) if cat_count and not people_count else None

        # keep only-dog samples (no people)
        dog_samples.append(str(annot_file)) if dog_count and not people_count else None

        # categorize current sample in terms of the annotated people
        if people_count >= extreme:
            many_people_samples["extreme"].append(str(annot_file))
        elif people_count >= difficult:
            many_people_samples["difficult"].append(str(annot_file))
        elif people_count >= medium:
            many_people_samples["medium"].append(str(annot_file))

    # random sample 200 samples from each person difficulty group
    num_samples = 100
    people_samples_balanced = []
    for sub_group in many_people_samples.values():
        people_samples_balanced.extend(random.sample(sub_group, min(num_samples, len(sub_group))))

    # random sample 5000 samples from each pet category
    num_samples_cat = 2700
    num_samples_dog = 2700
    cat_samples_balanced = []
    dog_samples_balanced = []
    cat_samples_balanced.extend(random.sample(cat_samples, min(num_samples_cat, len(cat_samples))))
    dog_samples_balanced.extend(random.sample(dog_samples, min(num_samples_dog, len(dog_samples))))
    pet_samples_balanced = cat_samples_balanced + dog_samples_balanced

    # count the number of cats, dogs and persons in current people samples
    cat_cnt, dog_cnt, people_cnt = 0, 0, 0
    for annot_file in people_samples_balanced:
        # get labels from the annotation file of current sample
        cls_ids, labels = get_labels(annot_file, classes)

        # count the occurrences of each class in current sample
        cat_cnt += labels.count("cat")
        dog_cnt += labels.count("dog")
        people_cnt += labels.count("person")

    print(f"People samples:\nCats: {cat_cnt} | Dogs: {dog_cnt} | People: {people_cnt}")
    total_cat_cnt, total_dog_cnt, total_people_cnt = cat_cnt, dog_cnt, people_cnt

    # count the number of cats, dogs and persons in current pet samples
    cat_cnt, dog_cnt, people_cnt = 0, 0, 0
    for annot_file in pet_samples_balanced:
        # get labels from the annotation file of current sample
        cls_ids, labels = get_labels(annot_file, classes)

        # count the occurrences of each class in current sample
        cat_cnt += labels.count("cat")
        dog_cnt += labels.count("dog")
        people_cnt += labels.count("person")

    print(f"\nPet samples:\nCats: {cat_cnt} | Dogs: {dog_cnt} | People: {people_cnt}")
    total_cat_cnt += cat_cnt
    total_dog_cnt += dog_cnt
    total_people_cnt += people_cnt

    # final summary
    print(f"\nTotal samples:\nCats: {total_cat_cnt} | Dogs: {total_dog_cnt} | People: {total_people_cnt}")

    # create dataset
    output_dataset = Path("prosegur_person_pet_balanced")
    images_path, labels_path = (output_dataset / "images" / split), (output_dataset / "labels" / split)
    images_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)
    orig_images_path = Path("/home/danigarciaoca/Imágenes/prosegur_pet-person_clean-merged_coco") / split / "images"
    total_samples = set(cat_samples_balanced + dog_samples_balanced + people_samples_balanced)
    for annot_file in total_samples:
        # copy image
        basename = annot_file.split("/")[-1].split(".")[0]
        src_img_path = orig_images_path / (basename + ".jpg")
        dst_img_path = images_path / (basename + ".jpg")
        shutil.copy(src_img_path, dst_img_path)

        # copy annotation file
        src_lbl_path = annot_file
        dst_lbl_path = labels_path / (basename + ".txt")
        shutil.copy(src_lbl_path, dst_lbl_path)

    print("Done!")
