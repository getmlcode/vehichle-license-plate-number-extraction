from tabnanny import verbose
import cv2
from shapely import box
from sympy import false
from tqdm import tqdm
from ultralytics import YOLO
import os
import pytesseract as pt
import re


def clean_image(image):
    # image increase contrast/brightness
    alpha = 1.2  # Contrast control (1.0-3.0)
    beta = 0.5  # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def get_license_plate_coordinates(model_path, img_path, isSave=False):
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
    return results[0].boxes.xyxy.int().tolist()


def get_license_plate_number_img(model_path, img_path, isSave=False):
    license_plates_coordinates = get_license_plate_coordinates(model_path, img_path)

    # OCR with tesseract
    img = cv2.imread(img_path)
    number_coord_and_text_list = []
    if license_plates_coordinates is not None:
        for coordinates in license_plates_coordinates:

            crop_obj = img[
                coordinates[1] : coordinates[3],
                coordinates[0] : coordinates[2],
            ]

            cleaned_img = clean_image(crop_obj)

            vehicle_number = pt.image_to_string(cleaned_img)
            vehicle_number = re.sub("[^0-9a-zA-Z]+", "", vehicle_number)

            number_coord_and_text_list.append((coordinates, vehicle_number))

        return img, number_coord_and_text_list


def detect_license_plate_number_vid(model_path, vid_path, out_vid_path, isSave=False):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(vid_path)
    w, h, fps = (
        int(cap.get(x))
        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    )
    video_writer = cv2.VideoWriter(
        out_vid_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame in tqdm(range(num_frames), total=num_frames):
        ret, img = cap.read()  # ret is True untill we reach the end of the video
        if ret == False:
            break

        results = model.predict(img, show=False, verbose=False)
        plate_coordinates = results[0].boxes.xyxy.int().tolist()
        # print(plate_coordinates)
        if len(plate_coordinates) > 0:
            for plate_coord in plate_coordinates:
                plate_top_left_X = plate_coord[0]
                plate_top_left_Y = plate_coord[1]
                plate_bottom_right_X = plate_coord[2]
                plate_bottom_right_Y = plate_coord[3]
                color = (255, 0, 0)
                cv2.rectangle(
                    img,
                    (plate_top_left_X, plate_top_left_Y),
                    (plate_bottom_right_X, plate_bottom_right_Y),
                    color,
                    2,
                )

                crop_obj = img[
                    plate_coord[1] : plate_coord[3], plate_coord[0] : plate_coord[2]
                ]
                cleaned_img = clean_image(crop_obj)
                vehicle_number = pt.image_to_string(cleaned_img)
                vehicle_number = re.sub("[^0-9a-zA-Z]+", "", vehicle_number)
                pos = (plate_top_left_X - 10, plate_top_left_Y - 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(
                    img,
                    vehicle_number,
                    pos,
                    font,
                    fontScale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
                video_writer.write(img)

    cap.release()
    video_writer.release()
