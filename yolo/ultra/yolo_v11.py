from ultralytics import YOLO
import os
import cv2
import pytesseract as pt
import re


def train(yaml_path, epochs_count, img_sz, device):
    if not os.path.exists(yaml_path):
        print("yaml doesn't exist")
        exit(0)

    model = YOLO("yolo11n.pt")
    model.train(
        data=yaml_path,  # path to dataset YAML
        epochs=epochs_count,
        imgsz=img_sz,
        device=device,
    )


def infer_image(model_path, img_path, isSave=False):
    if not os.path.exists(model_path):
        print("model path doesn't exist")
        exit(0)
    if not os.path.isfile(model_path):
        print("model path is not a file")
        exit(0)
    if not os.path.exists(img_path):
        print("image doesn't exist")
        exit(0)
    if not os.path.isfile(img_path):
        print("image path is not a file")
        exit(0)

    model = YOLO(model_path)
    results = model(img_path, save=isSave)
    results[0].show()


def infer_images(model_path, test_images_directory):
    if not os.path.exists(model_path):
        print("model_path doesn't exist")
        exit(0)
    if not os.path.isfile(model_path):
        print("model_path is not a file")
        exit(0)
    if not os.path.exists(test_images_directory):
        print("test_images_directory path provided doesn't exist")
        exit(0)
    if not os.path.isdir(test_images_directory):
        print("test_images_directory path provided is not a directory")
        exit(0)

    model = YOLO(model_path)
    results = model(test_images_directory, save=True)
