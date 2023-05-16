import shutil
import os

def generate_captions(image_class):
    return image_class


# Define the destination path
dst_folder = '<path to folder name you want the images to be copied>/images'


def generate_image_path(rootpath):
    """
    Copies images from subfolders within the given root path to a destination folder,
    generating unique image names within the limit of 255 characters.

    Args:
        rootpath (str): The root path containing subfolders with images.
    
    Description:
        The function iterates through each subfolder within the root path and generates
        a unique image name for each image within the subfolders. It then copies the images
        to a destination folder while ensuring the generated image names are within the
        255 character limit.

        The function excludes images with extensions such as '.svg' and filenames containing
        the word 'checkpoint'.
    """
    folder_list = os.listdir(rootpath)
    s = 0
    for folder in folder_list:
        if len(folder) == 7:
            curr_folder = rootpath + '/' + folder
            classes = os.listdir(curr_folder)
            for c in classes:
                curr_path = curr_folder + '/' + c
                for x, y, files in os.walk(curr_path):
                    i = 0
                    for f in files:
                        image_path = curr_path + '/' + f
                        extension = f.split('.')[-1]
                        if extension != 'svg' and 'checkpoint' not in f: 
                            image_name = c + '_' + str(i) + '.' + extension
                            shutil.copy(image_path, dst_folder + '/' + image_name)
                            i += 1
                    s += i
    print('count: ', s)

        
generate_image_path('<path to folder containing subfolders of images>')
img_folder = dst_folder

# create a file called captions.txt which contains index, image path and caption for that image (used in CLIP)
with open('<path to the folder you want the captions to be stored>/captions.txt', mode='w') as file:
# Write some text to the file
    file.write('index' + ","+ 'image'+ "," + "caption" + "\n")
    i = 1
    for x, y, images in os.walk(img_folder):
        for image in images:
            img_class = image.split('_')[0]
            caption  = generate_captions(img_class)
            file.writelines(str(i)+","+image+","+caption+"\n")
            i += 1