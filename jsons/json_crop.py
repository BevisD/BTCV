import json
import os
import random


def main():
    root_path = "/bask/projects/p/phwq4930-gbm/Bevis/NeOv_omentum_cropped"
    json_file = "neov-seg-cropped.json"

    num_train = 158
    num_test = 20
    num_val = 20

    description = "neov segmentation cropped"

    filenames = os.listdir(os.path.join(root_path, "pre_treatment", "images"))

    pairs = []
    for filename in filenames:
        pre_image_path = os.path.join("pre_treatment", "images", filename)
        pre_label_path = os.path.join("pre_treatment", "labels", filename)

        post_image_path = os.path.join("post_treatment", "images", filename)
        post_label_path = os.path.join("post_treatment", "labels", filename)

        pairs.append({"image": pre_image_path, "label": pre_label_path})
        pairs.append({"image": post_image_path, "label": post_label_path})

    random.shuffle(pairs)

    train_list = pairs[:num_train]
    test_list = pairs[num_train:num_train+num_test]
    val_list = pairs[num_train+num_test:num_train+num_test+num_val]

    json_dict = {"description": description,
                 "numTrain": len(train_list),
                 "numTest": len(test_list),
                 "numValidate": len(val_list),
                 "training": train_list,
                 "test": test_list,
                 "validation": val_list
                 }

    with open(os.path.join(root_path, json_file), "w+") as file:
        json.dump(json_dict, file, indent=2)


if __name__ == '__main__':
    main()
