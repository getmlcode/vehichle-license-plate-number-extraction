from yolo.ultra import yolo_v11
from vehicle_parts_detection import license_plate_detection
import cv2


if __name__ == "__main__":

    # yolo_v11.train(
    #     "D:\\DataSets\\ObejectDetection\\yolo11\\license_plate\\data.yaml",
    #     5,
    #     640,
    #     "cpu",
    # )

    # yolo_v11.infer_image(
    #     "D:\Acads\IISc ME\Projects\DLPretrainedModels\YoloUltralytics\LicensePlateFineTuned.pt",
    #     "D:\Acads\IISc ME\Projects\ObjectDetection\\test_images\\2.jpg",
    # )

    img, number_coords_and_text_list = (
        license_plate_detection.get_license_plate_number_img(
            "D:\Acads\IISc ME\Projects\DLPretrainedModels\YoloUltralytics\LicensePlateFineTuned.pt",
            "D:\Acads\IISc ME\Projects\ObjectDetection\\test_images\\11.jpg",
        )
    )

    if len(number_coords_and_text_list) > 0:
        for number_coords_and_text in number_coords_and_text_list:
            number_coords = number_coords_and_text[0]
            plate_top_left_X = number_coords[0]
            plate_top_left_Y = number_coords[1]
            plate_bottom_right_X = number_coords[2]
            plate_bottom_right_Y = number_coords[3]

            vehicle_number = number_coords_and_text[1]
            pos = (
                plate_top_left_X - 10,
                plate_top_left_Y - 10,
            )
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
            cv2.rectangle(
                img,
                (plate_top_left_X, plate_top_left_Y),
                (plate_bottom_right_X, plate_bottom_right_Y),
                color,
                2,
            )

    cv2.imshow("vehicle number", img)
    cv2.waitKey(0)

    cv2.imwrite(
        "D:\Acads\IISc ME\Projects\ObjectDetection\\test_images\\11_out.jpg", img
    )

    # license_plate_detection.detect_license_plate_number_vid(
    #     "D:\Acads\IISc ME\Projects\DLPretrainedModels\YoloUltralytics\LicensePlateFineTuned.pt",
    #     "D:\\Acads\\IISc ME\\Projects\\ObjectDetection\\test_vids\\2.mp4",
    #     "D:\\Acads\\IISc ME\\Projects\\ObjectDetection\\test_vids\\out_2.mp4",
    # )
