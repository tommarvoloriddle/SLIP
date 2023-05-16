import numpy as np
import os
import pandas as pd
from PIL import Image
import cv2

image_path = '<path to Evaluation dataset>'
destination_path = '<path to directory where you want your evaluation dataset images to be stored>'
num_output_images = 100

# Create a list of all the classes in the dataset
classes = os.listdir(image_path)

df = {
    'classes' : [],
    'images' : []
}

# pick an image for 4 random classes
# and combine 4 random images into a single image
base_image = np.zeros((224, 224, 3))

for j in range(50, num_output_images):
    # pick a random class
    stiched_class = ''
    class_ = 'pokemon-b'
        
    images = os.listdir(image_path + class_ + '/')
    random_image = np.random.choice(images, size=1, replace=False)
    image1 = cv2.imread(image_path + class_ +  '/' + random_image[0])
    stiched_class += class_ + '/' + random_image[0] + '_'
    # pick a random class
    # class_ = np.random.choice(classes)
    images = os.listdir(image_path + class_ + '/')
    random_image = np.random.choice(images, size=1, replace=False)
    image2 = cv2.imread(image_path + class_ +  '/' + random_image[0])
    stiched_class += class_ + '/' + random_image[0] + '_'
    # pick a random class
    # class_ = np.random.choice(classes)
    images = os.listdir(image_path + class_ + '/')
    random_image = np.random.choice(images, size=1, replace=False)
    image3 = cv2.imread(image_path + class_ +  '/' + random_image[0])
    stiched_class += class_ + '/' + random_image[0] + '_'
    # pick a random class
    # class_ = np.random.choice(classes)
    images = os.listdir(image_path + class_ + '/')
    random_image = np.random.choice(images, size=1, replace=False)
    image4 = cv2.imread(image_path + class_ +  '/' + random_image[0])
    stiched_class += class_ + '/' + random_image[0] + '_'

    if image1 is None or image2 is None or image3 is None or image4 is None:
        continue

    size = [224*2, 224*2]

    image1 = cv2.resize(image1, (size[0], size[1]))
    image2 = cv2.resize(image2, (size[0], size[1]))
    image3 = cv2.resize(image3, (size[0], size[1]))
    image4 = cv2.resize(image4, (size[0], size[1]))



    # Create a new image with the calculated dimensions
    grid = np.zeros((2*size[1], 2*size[0], 3), np.uint8)

    grid[:size[1], :size[0]] = image1
    grid[:size[1], size[0]:] = image2
    grid[size[1]:, :size[0]] = image3
    grid[size[1]:, size[0]:] = image4

    image_name = str(j)

    # Save the combined image
    cv2.imwrite(destination_path + image_name + ".png", grid)

    df['classes'].append(stiched_class)
    df['images'].append(image_name + ".png")

# create a sample DataFrame
data = pd.DataFrame(df)
df = pd.read_csv("<path to your evaluation csv file>")
# append the new data to the existing DataFrame
df = df._append(data, ignore_index=True)
# save the DataFrame to a CSV file
df.to_csv('<path to your evaluation csv file>', index=False)
