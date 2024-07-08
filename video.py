import cv2
import numpy as np

global out

def open_video(height: int, width: int, fps: int):
    global out
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
def write_frame(frame: np.array, fps: float, speed: float, max_speed: float,
                 min_speed: float, max_fps: float, min_fps: float):
    global out

    # calculate the % of max speed (0 - 1)
    speed_percentage = 1 - ((speed - min_speed) / (max_speed - min_speed))

    delta_fps = max_fps - min_fps

    # calculate the number of frames between min_fps and max_fps proportional to the speed between min_speed and max_speed
    nb_frames = int(min_fps + (speed_percentage * delta_fps))

    if nb_frames < min_fps:
        nb_frames = min_fps
    for i in range(nb_frames):
        out.write(frame)

def close_video():
    global out
    out.release()
    