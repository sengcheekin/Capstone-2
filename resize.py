# Code to resize and convert images to JPEG format

import os
from PIL import Image

def resize_and_convert_images(folder_path, output_size):
    # Create a new folder for resized images
    resized_folder_path = "D:\Documents\Semester 9\Capstone 2\Code\Capstone-2\datasets\o-haze\hazy\jpg"
    os.makedirs(resized_folder_path, exist_ok=True)

    # Iterate over the images in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Open the image
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)

            # Resize the image
            resized_image = image.resize(output_size)

            # Convert to JPEG format
            resized_image = resized_image.convert('RGB')

            # Save the resized image as JPEG
            resized_image_path = os.path.join(resized_folder_path, filename.split('.')[0] + '.jpg')
            resized_image.save(resized_image_path, 'JPEG')

            print(f"Resized and converted image saved at: {resized_image_path}")

    print("Image resizing and conversion complete!")

if __name__ == "__main__":
    # Path to the folder with images
    folder_path = 'D:\Documents\Semester 9\Capstone 2\Code\Capstone-2\datasets\o-haze\hazy'
    output_size = (256, 256)

    resize_and_convert_images(folder_path, output_size)
