import os
import csv

import nibabel as nib
import numpy as np


def main():
    PRE_PATH = "/bask/projects/p/phwq4930-gbm/Ines/Ovarian/Data/NeOv/pre_treatment/segmentations/"
    POST_PATH = "/bask/projects/p/phwq4930-gbm/Ines/Ovarian/Data/NeOv/post_treatment/segmentations/"
    CSV_FILE = "/bask/projects/p/phwq4930-gbm/Ines/Ovarian/Data/NeOv/labels.csv"

    pre_count = {i: 0 for i in range(19)}
    post_count = {i: 0 for i in range(19)}
    inter_count = {i: 0 for i in range(19)}

    with open(CSV_FILE, "r") as file:
        reader = csv.reader(file)
        header = next(reader)

        for i, row in enumerate(reader):
            print(f"Patient {i+1}/99")
            data = dict(zip(header, row))
            pre_file = data["pre_treatment"]
            post_file = data["post_treatment"]

            pre_seg = nib.load(os.path.join(PRE_PATH, pre_file)).get_fdata()
            post_seg = nib.load(os.path.join(POST_PATH, post_file)).get_fdata()

            pre_unique = set(pre_seg.ravel())
            post_unique = set(post_seg.ravel())

            inter = set.intersection(pre_unique, post_unique)

            for num in pre_unique:
                pre_count[num] += 1

            for num in post_unique:
                post_count[num] += 1

            for num in inter:
                inter_count[num] += 1

    with open("proportions.csv", "w+") as file:
        fieldnames = ["site-number", "pre-count", "post-count", "both-count"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        for num in range(19):
            writer.writerow({"site-number": num,
                             "pre-count": pre_count[num],
                             "post-count": post_count[num],
                             "both-count": inter_count[num]
                             })


if __name__ == '__main__':
    main()
