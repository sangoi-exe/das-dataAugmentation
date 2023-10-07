import os
import sys
import random
import cv2
import numpy as np
import tkinter as tk
from glob import glob
from tkinter import filedialog
from tqdm import tqdm
from PIL import Image, ExifTags

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

def rotate_image(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return result

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    return cv2.addWeighted(image, 1 + float(contrast) / 100.0, image, 0, float(brightness))

def adjust_saturation(image, saturation_factor):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] * saturation_factor, 0, 255)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def display_menu(input_dir, output_dir):
    print("MENU")
    print("1. Set input directory" + (f" [{input_dir}]" if input_dir else ""))
    print("2. Set output directory" + (f" [{output_dir}]" if output_dir else ""))
    print("3. Rotate augmentation")
    print("4. Contrast augmentation")
    print("5. Brightness augmentation")
    print("6. Saturation augmentation")
    print("7. Fix image orientation")
    print("8. Convert images to PNG")
    print("9. Resize images")
    print("0. Exit")
    return int(input("Choose an option: "))

def select_directory(title):
    root = tk.Tk()
    root.withdraw()
    print(f"Please select the {title} directory...")
    return filedialog.askdirectory(title=f"Select {title} directory")

def next_filename(basename, ext):
    if '(' in basename and ')' in basename:
        name, num = basename.rsplit(' (', 1)
        num = num.rstrip(')')
        if num.isdigit():
            return f"{name} ({int(num) + 1}){ext}"
    return f"{basename} (1){ext}"

def generate_unique_filename(output_dir, basename, ext):
    filename = next_filename(basename, ext)
    while os.path.exists(os.path.join(output_dir, filename)):
        filename = next_filename(basename, ext)
    return filename

def process_images(input_dir, output_dir, option, num_variants, include_original):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_formats = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff')
    image_paths = []

    for format in image_formats:
        image_paths.extend(glob(os.path.join(input_dir, format)))

    for image_path in tqdm(image_paths, desc="Processing images", ncols=100, unit="image"):
        image = cv2.imread(image_path)
        basename, ext = os.path.splitext(os.path.basename(image_path))

        if include_original:
            unique_original_filename = generate_unique_filename(output_dir, basename, ext)
            original_image_path = os.path.join(output_dir, unique_original_filename)
            cv2.imwrite(original_image_path, image)

        for i in range(num_variants):
            if option == 3:
                angle = random.uniform(-5, 5)
                modified_image = rotate_image(image, angle)
            elif option == 4:
                contrast = random.uniform(0.8, 1.2)
                modified_image = adjust_brightness_contrast(image, contrast=contrast)
            elif option == 5:
                brightness = random.uniform(-50, 50)
                modified_image = adjust_brightness_contrast(image, brightness=brightness)
            elif option == 6:
                saturation_factor = random.uniform(0.8, 1.2)
                modified_image = adjust_saturation(image, saturation_factor)

            unique_modified_filename = generate_unique_filename(output_dir, basename, ext)
            modified_image_path = os.path.join(output_dir, unique_modified_filename)
            cv2.imwrite(modified_image_path, modified_image)

    print("Data augmentation completed.")

def fix_orientation(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        basename, ext = os.path.splitext(filename)
        if ext.lower() in IMAGE_EXTENSIONS:
            img = Image.open(os.path.join(input_dir, filename))
            img = correct_orientation(img)
            img.save(os.path.join(output_dir, f"{basename}.png"), "PNG")

def correct_orientation(img):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(img._getexif().items())
        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return img

def convert_to_png(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        basename, ext = os.path.splitext(filename)
        if ext.lower() in IMAGE_EXTENSIONS:
            img = Image.open(os.path.join(input_dir, filename))
            img = correct_orientation(img)
            img.save(os.path.join(output_dir, f"{basename}.png"), "PNG")

def resize_images(input_dir, output_dir, size):
    for filename in os.listdir(input_dir):
        basename, ext = os.path.splitext(filename)
        if ext.lower() in IMAGE_EXTENSIONS:
            img = Image.open(os.path.join(input_dir, filename))
            img = resize_image(img, size)
            img.save(os.path.join(output_dir, f"{basename}.png"), "PNG")

def resize_image(img, size=(512, 512)):
    img.thumbnail(size, Image.ANTIALIAS)
    width, height = img.size
    new_img = Image.new("RGBA", size, (0, 0, 0, 0))
    new_img.paste(img, ((size[0] - width) // 2, (size[1] - height) // 2))
    return new_img

def ask_include_original():
    while True:
        choice = input("Include original images in the output folder? (y/n): ").lower()
        if choice == 'y':
            return True
        elif choice == 'n':
            return False
        else:
            print("Invalid option. Please enter 'y' or 'n'.")

def ask_number_of_variants():
    while True:
        try:
            num_variants = int(input("Enter the number of variants for each image: "))
            if num_variants > 0:
                return num_variants
            else:
                print("Invalid number. Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a positive integer.")

def main():
    input_dir = ""
    output_dir = ""
    include_original = False
    num_variants = 0
    os.system("cls" if os.name == "nt" else "clear")

    while True:
        option = display_menu(input_dir, output_dir)

        if option == 1:
            input_dir = select_directory("Input")
        elif option == 2:
            output_dir = select_directory("Output")
        elif option in (3, 4, 5, 6):
            if not (input_dir and output_dir):
                print("Please set both input and output directories.")
                continue

            include_original = ask_include_original()
            num_variants = ask_number_of_variants()

            process_images(input_dir, output_dir, option, num_variants, include_original)
            print("Image augmentation completed.")
        elif option == 7:
            if not (input_dir and output_dir):
                print("Please set both input and output directories.")
                continue

            fix_orientation(input_dir, output_dir)
            print("Orientation fix completed.")
        elif option == 8:
            if not (input_dir and output_dir):
                print("Please set both input and output directories.")
                continue

            convert_to_png(input_dir, output_dir)
            print("PNG conversion completed.")
        elif option == 9:
            if not (input_dir and output_dir):
                print("Please set both input and output directories.")
                continue

            size_option = int(input("Select the image size:\n1. 512x512\n2. 768x768\n3. 1024x1024\nChoose an option: "))
            size = {1: (512, 512), 2: (768, 768), 3: (1024, 1024)}[size_option]

            resize_images(input_dir, output_dir, size)
            print("Image resizing completed.")
        elif option == 0:
            break
        else:
            print("Invalid option. Please choose an option from the menu.")

if __name__ == "__main__":
    main()