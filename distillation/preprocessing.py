from PIL import Image
import numpy as np
from torchvision import transforms
import cv2

def vertical_crop(img, crop_top=0, crop_bottom=0):
    width, height = img.size
    img_cropped = img.crop((0, crop_top, width, height - crop_bottom))
    return img_cropped

def horizontal_crop(img, crop_left=0, crop_right=0):
    width, height = img.size
    img_cropped = img.crop((crop_left, 0, width - crop_right, height))
    return img_cropped

def isolate_and_grayscale(img: Image.Image) -> Image.Image:
    img_np = np.array(img.convert("RGB"))
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    lower_brown = np.array([10, 100, 20])
    upper_brown = np.array([30, 255, 200])
    mask_isolate_brown = cv2.inRange(img_hsv, lower_brown, upper_brown)
    img_isolated = cv2.bitwise_and(img_np, img_np, mask=mask_isolate_brown)
    img_gray = cv2.cvtColor(img_isolated, cv2.COLOR_RGB2GRAY)
    return Image.fromarray(img_gray, mode='L')

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(lambda img: vertical_crop(img, crop_top=125, crop_bottom=55)), 
    transforms.Lambda(lambda img: horizontal_crop(img, crop_left=90, crop_right=90)), 
    transforms.Lambda(lambda img: isolate_and_grayscale(img)),
    transforms.Resize((84,84)),  
    transforms.ToTensor()
])
