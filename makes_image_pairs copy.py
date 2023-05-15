import numpy as np
import os
from PIL import Image
image_path = '/scratch/sg7729/DL_project/SLIP/Pokemon Dataset/'
destination_path = '/scratch/sg7729/DL_project/SLIP/pairs/'
Num_of_pairs = 10

# Create a list of all the classes in the dataset
classes = os.listdir(image_path)

# pick an image for 4 random classes
# and combine 4 random images into a single image
base_image = np.zeros((224, 224, 3))
for j in range(Num_of_pairs):
    # pick a random class
    class_ = ''
    # pick 2 random images from the class
    images = os.listdir(image_path)
    random_image = np.random.choice(images, size=1, replace=False)
    print(random_image)
    # load the image
    image1 = Image.open(image_path  + '/' + random_image[0])

    # pick 2nd random class
    class_ = np.random.choice(classes)
    # pick 2 random images from the class
    images = os.listdir(image_path)
    random_image = np.random.choice(images, size=1, replace=False)
    print(random_image)
    # load the image
    image2 = Image.open(image_path  + '/' + random_image[0])

    # pick 3rd random class
    class_ = np.random.choice(classes)
    # pick 2 random images from the class
    images = os.listdir(image_path)
    random_image = np.random.choice(images, size=1, replace=False)
    print(random_image)
    # load the image
    image3 = Image.open(image_path  + '/' + random_image[0])

    # pick 4th random class
    class_ = np.random.choice(classes)
    # pick 2 random images from the class
    images = os.listdir(image_path)
    random_image = np.random.choice(images, size=1, replace=False)
    print(random_image)
    # load the image
    image4 = Image.open(image_path  + '/' + random_image[0])

    # Get the dimensions of the images
    width1, height1 = image1.size
    width2, height2 = image2.size
    width3, height3 = image3.size
    width4, height4 = image4.size

    # Calculate the dimensions of the new image
    new_width = max(width1, width3) + max(width2, width4)
    new_height = max(height1, height2) + max(height3, height4)

    # Create a new image with the calculated dimensions
    new_image = Image.new('RGB', (new_width, new_height))

    # Paste the first image in the top left corner
    new_image.paste(image1, (0, 0))

    # Paste the second image in the top right corner
    new_image.paste(image2, (width1, 0))

    # Paste the third image in the bottom left corner
    new_image.paste(image3, (0, height1))

    # Paste the fourth image in the bottom right corner
    new_image.paste(image4, (width3, height1))

    image_name = str(j)

    # Save the combined image
    new_image.save(destination_path + image_name + ".jpg")

