# Ball Tracking with YOLO and Interpolation

This project demonstrates how to track a ball in a video using a custom YOLO object detection framework trained for ball detection and interpolation to handle areas where the tracking fails.

![input_video1 - frame at 0m0s](https://github.com/user-attachments/assets/f6db1088-ff93-4957-899a-23cbdf407148)
*A frame from the Output*

## Key Features

* **Training:** Train your YOLO model for optimal ball detection in your specific scenario using Roboflow. 
* **YOLO Object Detection:** Use the trained YOLO model for real-time and accurate ball detection within video frames.
* **Interpolation:** Handles missing ball detections (e.g. due to occlusion) by interpolating the ball's position based on previous and subsequent detections.
* **Visualization:** Draws bounding boxes around detected balls and displays frame numbers on the output video for analysis.


https://github.com/user-attachments/assets/41a16a84-4800-4f29-847f-3df619fd42da

*Output without interpolation*

This output is unable to track the ball during some of the instances like when the ball crosses the net. To solve this we created the interpolation function, that the track the ball in this instances by interpolating the current position using the previous and next detected positions.

## Installation and Training

1. **Install necessary libraries:**
   ```bash
   pip install ultralytics opencv-python pandas roboflow
   ```

2. **Create a Roboflow Account:** We need this to manage our computer vision datasets and training.

```
project = Roboflow(api_key="your_API_key").workspace("viren-dhanwani").project("tennis-ball-detection")
version = project.version(6)
dataset = version.download("yolov8")
```

3. **Connect your Roboflow project:** Enter your Roboflow API key by replacing `"your_API_key"` (available in your account settings)

4. **Download the dataset:** The code downloads the dataset version 6 from the Roboflow project.

5. **Train the YOLO model:** Training the model using the downloaded dataset and necessary:

```
from ultralytics import YOLO

!yolo task=detect mode=train model=yolov8x.pt data=tennis-ball-detection-6/data.yaml epochs=150 imgsz=640

```

## Use of the custom YOLO model

Place the model file in the models directory.

1. **Prepare the input video:** Place your input video file in the input_files directory.

2. **Run the script:** Execute the predict.py script.

3. **Output:** The processed video with bounding boxes and frame numbers will be saved in the saved_outputs directory.


## Everything in a Nutshell

The program reads the video and extracts individual frames using function `read_video`. It creates a `BallTracker` object with the YOLO model path for ball detection. Each frame is passed to the `BallTracker` for object detection using `detect_frames`. The detected ball's bounding box coordinates are stored for each frame. The function `interpolate_ball_positions` handles frames where the ball might be missed by the custom YOLO model. Bounding boxes and frame numbers are visualized on each frame by `draw_bboxes`. Finally, `save_video` processes the video with visualizations and saves it as a new file.

https://github.com/user-attachments/assets/1a86e148-3e00-45fa-94d2-d6a9aa6d14c0

*Final output video*

## Contributions

Contributions are welcome! Feel free to fork this repository, make improvements, and submit pull requests.

## License

This project is licensed under the MIT License.


 
