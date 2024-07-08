import cv2
import numpy as np
import math
import time

import funcs
import video

NUMBER_OF_COLS: int = 8 #number of solenoids
ROW_PX_FROM_TOP: int = 200 #detection zone for solenoids
THRESHOLD: int = 30 #in % of green in the column
MAX_SPEED: float = 2.5 #in km/s simaled
MIN_SPEED: float = 0.5 #in km/s simaled
OUTPUT_FPS: float = 20 #fps of the output video
MIN_FPS: int = 1 #minimun number of frames per frame
MAX_FPS: int = 5 #maximun number of frames per frame

PLANNING: list = [
    {
        "timestamp": 18,
        "rotation": 270
    },
    {
        "timestamp": 19,
        "rotation": 180
    },
    {
        "timestamp": 100,
        "rotation": 90
    }
]

source_fps: int = 10
current_frame: int = 1

def analyze_frame(cap):
    global current_frame

    # Get the start time of current frame
    start_frame = time.time()

    # declare array of false for solenoid sim
    solenoid_active = funcs.declare_solenoid_active(NUMBER_OF_COLS)

    # Get the frame from the video
    frame = funcs.get_frame(cap)

    #rotate the frame according to the planning
    current_frame += 1
    frame = funcs.rotate(frame, current_frame, source_fps, PLANNING)

    # get the excess of green 
    exg = funcs.calcluate_exg(frame)

    # threshold the image
    tresh = funcs.threshold(exg)

    # open and close the image
    opened_closed = funcs.open_and_close_image(tresh)

    # color the detected green on the original frame
    frame[opened_closed==255] = (0,255,0)

    #edges = funcs.detect_edges_canny(opened_closed)

    # get the active solenoids and the speed the robot should move
    solenoid_active = funcs.activate_solenoids(NUMBER_OF_COLS, ROW_PX_FROM_TOP,
                                                opened_closed, solenoid_active, 
                                                THRESHOLD)
    
    speed = funcs.get_speed(solenoid_active, MAX_SPEED, MIN_SPEED)

    # calculate the fps
    end_frame = time.time()
    delta_time_frame = end_frame - start_frame
    fps = 1 / delta_time_frame

    # print the UI on the frame
    result = funcs.printUI(frame, NUMBER_OF_COLS, ROW_PX_FROM_TOP,
                            solenoid_active, fps, speed)
    
    # add the frame to the output video
    video.write_frame(result, OUTPUT_FPS, speed, MAX_SPEED, MIN_SPEED, MAX_FPS, MIN_FPS)

    return result

def main():
    global source_fps
    cap = cv2.VideoCapture("test_data/video1.mp4")
    source_fps = int(cap.get(cv2.CAP_PROP_FPS))
    if not (cap.isOpened()):
        cap.release()
        cv2.destroyAllWindows()
        raise IOError("Could not open video device")
    else:
        # sample first frame to get the width and height for the output video
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video.open_video(width, height, OUTPUT_FPS)

    while cap.isOpened():
        try:
            result = analyze_frame(cap)
            cv2.imshow("Result", result)
        except Exception as e:
            print(e)
            video.close_video()
            break
        
        # press q to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            video.close_video()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()