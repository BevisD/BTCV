import nibabel as nib
import os


def copy_img(filename, from_dir, to_dir, sub_folder="images"):
    file = nib.load(os.path.join(from_dir, sub_folder, filename))

    data = file.get_fdata()
    pre, post = data

    pre_img = nib.Nifti1Image(pre, file.affine, file.header)
    post_img = nib.Nifti1Image(post, file.affine, file.header)

    nib.save(pre_img, os.path.join(to_dir, "pre_treatment", sub_folder, filename))
    nib.save(post_img, os.path.join(to_dir, "post_treatment", sub_folder, filename))


def main():
    from_dir = "/bask/projects/p/phwq4930-gbm/Thomas/miab_data_base/raw_data/NeOv_main_sites_cropped/"
    to_dir = "/bask/projects/p/phwq4930-gbm/Bevis/NeOv_omentum_cropped"

    image_names = os.listdir(os.path.join(from_dir, "images"))
    label_names = os.listdir(os.path.join(from_dir, "labels"))

    for i, image_name in enumerate(image_names):
        print(f"Copying image {image_name} {i}/{len(image_names)}")
        copy_img(filename=image_name,
                 from_dir=from_dir,
                 to_dir=to_dir,
                 sub_folder="images"
                 )

    for i, label_name in enumerate(label_names):
        print(f"Copying label {label_name} {i}/{len(label_names)}")
        copy_img(filename=label_name,
                 from_dir=from_dir,
                 to_dir=to_dir,
                 sub_folder="labels"
                 )


if __name__ == '__main__':
    main()
