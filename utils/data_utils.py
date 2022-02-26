import os
import numpy as np
import PIL
import imageio
import cv2
from imutils.video import count_frames
import torch
from utils.common import tensor2im


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

VIDEO_EXTENSIONS = [
    '.MOV', '.avi', '.gif', 'mp4'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VIDEO_EXTENSIONS)


def image_files(dir_):
    files = os.listdir(dir_)
    l = [os.path.join(dir_, file) for file in files if is_image_file(file)]
    return sorted(l)


def create_video(images, path, mode="cv2"):
    if mode == "imageio":
        imageio.mimsave(path, images)
    
    elif mode == "cv2":
        height, width, _ = np.array(images[0]).shape
        size = (width, height)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
        for image in images:
            array = np.array(image)
            array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            writer.write(array)
            
        writer.release()


def extract_frames(video_path, rotate=0, interval=(0, 1)):
    n_frames = count_frames(video_path, override=False)
    vidcap = cv2.VideoCapture(video_path)
    success, _ = vidcap.read()
    
    a = int(interval[0] * n_frames)
    b = int(interval[1] * n_frames)
    
    images = []
    
    for i in range(n_frames):
        success, array = vidcap.read()
        if success and i >= a and i <= b:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
            image = PIL.Image.fromarray(array).rotate(rotate)
            images.append(image)
    
    return images


def paste(image_0, image_1):
    dst = PIL.Image.new('RGB', (image_0.width + image_1.width, image_0.height))
    dst.paste(image_0, (0, 0))
    dst.paste(image_1, (image_0.width, 0))
    return dst


def resize_gif(path, size):
    images = extract_frames(path)
    r = []
    for image in images:
        r.append(image.resize(size))
    save_path = path[:-4] + "_.gif"
    create_video(r, save_path, mode="imageio")
    
    
def exif_transpose(image):
    try:
        image = PIL.ImageOps.exif_transpose(image)
    except:
        pass
    return image


def generate_faces(generator, n_faces):
    images = []
    for _ in range(n_faces):
        z = torch.randn(1, 512).cuda()
        x, _ = generator([z])
        image = tensor2im(x[0])
        images.append(image)
    return images