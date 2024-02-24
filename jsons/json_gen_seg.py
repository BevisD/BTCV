import json
import os
import csv
import random
import argparse
import configparser


# Parse the input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="JSON file creator from csv file")
    parser.add_argument("--csv-file", default="labels.csv", type=str, help="csv file containing patient data")
    parser.add_argument("--json-file", default="data.json", type=str, help="json file to store patient data")
    parser.add_argument("--data-dir", default="./", type=str, help="json file to store patient data")
    parser.add_argument("--shuffle", action="store_true", help="shuffle the data or not")
    parser.add_argument("--num-train", type=int, help="number of training datapoints", required=True)
    parser.add_argument("--num-test", type=int, help="number of testing datapoints", required=True)
    parser.add_argument("--data-name", default="neov", help="name of the dataset")

    args = parser.parse_args()
    return args


def main(args):

    csv_pth = os.path.join(args["data_dir"], args["csv_file"])
    json_pth = os.path.join(args["data_dir"], args["json_file"])

    train_num = args["num_train"]
    test_num = args["num_test"]

    json_dict = {}
    json_dict["description"] = args["data_name"]

    pre_img_seg_pairs = []
    post_img_seg_pairs = []

    with open(csv_pth, "r") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)

        for row in reader:
            info = dict(zip(header, row))

            pre_filename = info["pre_treatment"]
            post_filename = info["post_treatment"]

            # Assumes there are pre_treatment and post_treatment folders
            # Each folder has images and segmentations
            pre_img_path = os.path.join("pre_treatment/images", pre_filename)
            pre_seg_path = os.path.join("pre_treatment/segmentations", pre_filename)

            post_img_path = os.path.join("post_treatment/images", post_filename)
            post_seg_path = os.path.join("post_treatment/segmentations", post_filename)

            pre_img_seg_pairs.append((pre_img_path, pre_seg_path))
            post_img_seg_pairs.append((post_img_path, post_seg_path))

    img_seg_pairs = pre_img_seg_pairs + post_img_seg_pairs
    if args["shuffle"]:
        random.seed(0)
        random.shuffle(img_seg_pairs)

    train_pairs = img_seg_pairs[:train_num]
    test_pairs = img_seg_pairs[train_num:train_num + test_num]
    val_pairs = img_seg_pairs[train_num + test_num:]

    json_dict["numTrain"] = len(train_pairs)
    json_dict["numTest"] = len(test_pairs)
    json_dict["numValidate"] = len(val_pairs)

    json_dict["test"] = [{"image": pair[0], "label": pair[1]} for pair in test_pairs]
    json_dict["training"] = [{"image": pair[0], "label": pair[1]} for pair in train_pairs]
    json_dict["validation"] = [{"image": pair[0], "label": pair[1]} for pair in val_pairs]

    with open(json_pth, "w+") as json_file:
        json.dump(json_dict, json_file, indent=2)


if __name__ == '__main__':
    parsed_args = parse_args()
    main(vars(parsed_args))
