from PIL import Image
import os

source_folder = 'C:/Users/THANH/Desktop/Exercises/4/Web_Tech_and_Service/Web_Project/ct313h03-project-Tmaster1303/frontend/public/'
destination_folder = 'C:/Users/THANH/Desktop/Exercises/4/Web_Tech_and_Service/Web_Project/ct313h03-project-Tmaster1303/frontend/public/'

directory = os.listdir(source_folder)
print(directory)

for item in directory:
    img = Image.open(source_folder + item)
    width, height = img.size
    ratio = width / height
    new_width = 300
    new_height = 300
    imgResize = img.resize((new_width, new_height), Image.LANCZOS)
    imgResize.save(destination_folder + item[:-4] + '.png', quality=100)
