import shutil
import os

# rootpath = os.

def generate_captions(image_class):
    return "This is an image of "+image_class


# Define the source and destination paths
# src_file = '/path/to/source/file.txt'
# dst_file = '/path/to/destination/file.txt'
dst_folder = '/scratch/sa6981/Deep-Learning-Pokemon/images'


def generate_image_path(rootpath):
  folder_list = os.listdir(rootpath)
  for folder in folder_list:
    if len(folder) == 7:
        curr_folder = rootpath + '/' + folder
        classes = os.listdir(curr_folder)
        for c in classes:
            curr_path = curr_folder + '/' + c
            for x, y, files in os.walk(curr_path):
                for file in files:
                    image_path = curr_path + '/' + file
                    image_name = c + '_' + file
                    shutil.copy(image_path, dst_folder + '/' + image_name)
        
# generate_image_path('/scratch/sa6981/Deep-Learning-Pokemon')
img_folder = dst_folder

# if __name__==__main__:

with open('/scratch/sa6981/Deep-Learning-Pokemon/captions.txt', mode='w') as file:
# Write some text to the file
    file.write('index' + ","+ 'image'+ "," + "caption" + "\n")
    i = 1
    for x, y, images in os.walk(img_folder):
        for image in images:
            img_class = image.split('_')[0]
            caption  = generate_captions(img_class)
            # make_txt_file(str(i), image, caption)
            file.writelines(str(i)+","+image+","+caption+"\n")
            i += 1