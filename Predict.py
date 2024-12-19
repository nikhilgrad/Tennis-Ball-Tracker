#complete code integration
from ultralytics import YOLO
import cv2
import pandas as pd

class BallTracker:
    # ... (rest of the BallTracker class)

    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        ball_detections = []

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)
        
        return ball_detections

    def detect_frame(self,frame):
        results = self.model.predict(frame,conf=0.15)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames


def read_video(video_path):
    """Reads a video file and returns a list of frames.

    Args:
        video_path (str): Path to the video file.

    Returns:
        list: A list of frames, where each frame is a NumPy array.
    """

    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    #cv2.destroyAllWindows()

    return frames

def save_video(frames, output_path, fps=30):
    """Saves a list of frames as a video file.

    Args:
        frames (list): A list of frames, where each frame is a NumPy array.
        output_path (str): Path to the output video file.
        fps (int, optional): Frames per second of the output video. Defaults to 30.
    """

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()


def main():
    # Read Video
    input_video_path = 'input_files/input_video1.mp4'
    video_frames = read_video(input_video_path)

    # Create a BallTracker object
    ball_tracker = BallTracker(model_path='models/yolov8best.pt')

    # Detect and interpolate ball positions
    ball_detections = ball_tracker.detect_frames(video_frames)
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Draw bounding boxes and frame numbers
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the output video
    save_video(output_video_frames, "saved_outputs/output_video.avi")

if __name__ == "__main__":
    main()