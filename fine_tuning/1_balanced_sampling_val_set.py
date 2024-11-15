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
    split = "val"
    dataset_base_path = Path("prosegur_pet-person_clean-merged_yolo")
    annotations_path = dataset_base_path / "labels" / split

    classes = ["cat", "dog", "person"]

    # get those images with more than `num_people` people
    cat_samples = []
    dog_samples = []
    many_people_ranges = {
        "very_easy": range(1, 6),  # 1 to 5
        "easy": range(6, 11),  # 6 to 10
        "medium": range(11, 16),  # 11 to 15
        "difficult": range(16, 21),  # 16 to 20
        "extreme": range(21, 26)  # 21 to 25
    }
    many_people_samples = {"very_easy": [], "easy": [], "medium": [], "difficult": [], "extreme": []}
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
        for category, people_range in many_people_ranges.items():
            if people_count in people_range:
                many_people_samples[category].append(str(annot_file))
                break

    # random sample 200 samples from each person difficulty group
    many_people_num_samples = {"very_easy": 300, "easy": 70, "medium": 40, "difficult": 0, "extreme": 0}
    people_samples_balanced = []
    many_people_samples_aux = {"very_easy": 0, "easy": 0, "medium": 0, "difficult": 0, "extreme": 0}
    people_count_tot_aux = 0
    for dif_level, sub_group in many_people_samples.items():
        num_samples = many_people_num_samples[dif_level]
        sample_aux = random.sample(sub_group, num_samples)
        for annot_file in sample_aux:
            # get labels from the annotation file of current sample
            cls_ids, labels = get_labels(annot_file, classes)
            people_count = labels.count("person")
            # categorize current sample in terms of the annotated people
            for category, people_range in many_people_ranges.items():
                if people_count in people_range:
                    many_people_samples_aux[category] += people_count
                    people_count_tot_aux += people_count
                    break

        people_samples_balanced.extend(sample_aux)

    print(f" - People samples: {len(people_samples_balanced)} ({people_count_tot_aux} annots)")
    print(f"   - very easy: {many_people_num_samples['very_easy']} ({many_people_samples_aux['very_easy']} annots)")
    print(f"   - easy: {many_people_num_samples['easy']} ({many_people_samples_aux['easy']} annots)")
    print(f"   - medium: {many_people_num_samples['medium']} ({many_people_samples_aux['medium']} annots)")
    print(f"   - difficult: {many_people_num_samples['difficult']} ({many_people_samples_aux['difficult']} annots)")
    print(f"   - extreme: {many_people_num_samples['extreme']} ({many_people_samples_aux['extreme']} annots)")

    # random sample 5000 samples from each pet category
    num_samples_cat = 580
    num_samples_dog = 390
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

    print(f"\nPeople annots:\nCats: {cat_cnt} | Dogs: {dog_cnt} | People: {people_cnt}")
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

    print(f"\nPet annots:\nCats: {cat_cnt} | Dogs: {dog_cnt} | People: {people_cnt}")
    total_cat_cnt += cat_cnt
    total_dog_cnt += dog_cnt
    total_people_cnt += people_cnt

    # final summary
    print(f"\nTotal annots:\nCats: {total_cat_cnt} | Dogs: {total_dog_cnt} | People: {total_people_cnt}")

    # create dataset
    output_dataset = Path("prosegur_person_pet_balanced_2")
    images_path, labels_path = (output_dataset / "images" / split), (output_dataset / "labels" / split)
    images_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)
    orig_images_path = Path("/home/danigarciaoca/Imágenes/prosegur_pet-person_clean-merged_coco") / split / "images"
    total_samples = set(cat_samples_balanced + dog_samples_balanced + people_samples_balanced)

    print(f"\nTotal samples: {len(cat_samples_balanced) + len(dog_samples_balanced) + len(people_samples_balanced)}")
    print(f"Total unique samples: {len(total_samples)}")

    print("\nCreating dataset...")
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
