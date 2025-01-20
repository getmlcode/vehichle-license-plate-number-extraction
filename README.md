# Vehicle License Plate Number Extraction
Trained YOLOv11 on `custom dataset` of license plates with `ultralytics` and used tesseract to extract vehicle number from the license plate

#### *`Dataset`*
Download the License Plate Dataset from [here](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4)

#### *`Detection and Extraction On Video`*
shown as gif (github doesn't display videos)
![](test_vids/out_1.gif)

#### *`Detection and Extraction On Images`*
|               |		        |
| ------------- |:-------------:|
![](test_images/1_out.jpg) | ![](test_images/2_out.jpg) 
![](test_images/12_out.jpg)  | ![](test_images/14_out.jpg) 
![](test_images/4_out.jpg)  | ![](test_images/11_out.jpg) 

### *`Training Results By Ultralytics`*
I didn't have enough space in my C: to install CUDA and NVIDIA for some reason doesn't let you choose another drive.  
So I trained it on CPU for 5 Epochs and it took 22 hours.
|               |		        |
| ------------- |:-------------:|
![](runs/detect/train2/results.png) | ![](runs/detect/train2/F1_curve.png) 
![](runs/detect/train2/PR_curve.png)  | ![](runs/detect/train2/P_curve.png) 
![](runs/detect/train2/R_curve.png)  | ![](runs/detect/train2/labels_correlogram.png) 
![](runs/detect/train2/labels.png)  | ![](runs/detect/train2/confusion_matrix.png) 

# Future Work
* [ ] Add GUI.
* [ ] Add more object detection models other than YOLO.
* [ ] Live detection and extraction.

# Requirements
os  
ultralytics  
cv2  
re  
pytesseract  
tqdm

# References
Guided by this [Youtube Tutorial](https://www.youtube.com/watch?v=POmyidzahLg)  
