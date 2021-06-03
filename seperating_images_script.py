
# we have original images and masks path, make teo new folder namely test and trainset. Make images and masks inside testset folder, matched images will be stored there, and for train images and masks, copy the 880 images and masks that is remaining to the trainset folder
# this file takes the image and folder path files in original_images_folder, then it compares names with the test_split.csv file which has 120 images, 
# if the condition is true then, copies the images and masks from ori_mask_file_path to new_testset_mask_folder and deletes from the ori_mask_file_path
# then we should manually takes images and masks folder which has 880 images inside the new trainset folder

import csv
import os
import shutil

def seperate(csv_filepath, image_path):

    with open(csv_filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        new_testset_images_folder = os.path.join(image_path,"testset", "images")
        new_testset_mask_folder = os.path.join(image_path,"testset", "masks")

        #original path bata image extract
        original_images_folder = os.path.join(image_path,"images")

        images_names = os.listdir(original_images_folder)
        for row in csv_reader:

            for image_name in images_names:
                row_image_name = row[0]+".jpg"

                ori_image_file_path = os.path.join(image_path, "images", image_name)
                ori_mask_file_path = os.path.join(image_path, "masks", image_name)

                if row_image_name == image_name:

                    shutil.copy(ori_image_file_path, new_testset_images_folder)
                    shutil.copy(ori_mask_file_path, new_testset_mask_folder)
                    os.remove(ori_image_file_path)
                    os.remove(ori_mask_file_path)

seperate("/Users/shruti/NAAMIIProjects/POLYP_CCNET/dataset/test_split.csv", "/Users/shruti/NAAMIIProjects/POLYP_CCNET/dataset/Kvasir-SEG/")