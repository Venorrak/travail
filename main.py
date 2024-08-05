import cv2
import numpy as np
import math
import time

import funcs
import video

NUMBER_OF_COLS: int = 8 #number of solenoids

THRESHOLD: int = 30 #in % of green in the column
MAX_SPEED: float = 2.5 #in km/s simaled
MIN_SPEED: float = 0.5 #in km/s simaled
OUTPUT_FPS: float = 20 #fps of the output video
MIN_FPS: int = 1 #minimun number of frames per frame
MAX_FPS: int = 5 #maximun number of frames per frame


SIZE_FACTOR: float = 0.5 #factor to resize the frame
ROW_PX_FROM_TOP: int = int(200 * SIZE_FACTOR) #detection zone for solenoids
SPRAY_RANGE: int = int(250 * SIZE_FACTOR)  #range of the spray in px
FONT_SCALE = SIZE_FACTOR #scale of the font


SPRAY_INTENSITY: int = 100 #intensity of the spray 0-255
SPRAY_SPACING: int = int(40 * SIZE_FACTOR) #spacing between the spray


# params for corner detection 
FEATURE_PARAMS = dict( maxCorners = 100, 
                       qualityLevel = 0.3, 
                       minDistance = 7, 
                       blockSize = 7 ) 
  
# Parameters for lucas kanade optical flow
LK_PARAMS = dict( winSize = (15, 15), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) 

PLANNING: list = [
    {
        "timestamp": 18,
        "rotation": 270
    },
    {
        "timestamp": 100,
        "rotation": 90
    }
]

source_fps: int = 10
current_frame: int = 1
old_gray: np.array = None
p0 = None

def analyze_frame(cap):
    global current_frame
    # Get the start time of current frame
    start_frame = time.time()

    # declare array of false for solenoid sim
    solenoid_active = funcs.declare_solenoid_active(NUMBER_OF_COLS)

    # Get the frame from the video
    ret, frame_original = cap.read()

    #rotate the frame according to the planning
    current_frame += 1
    frame = funcs.rotate(frame_original, current_frame, source_fps, PLANNING)
    
    ##################
    ## optical flow ##
    ##################
    
    #calculate the optical flow
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # calculate optical flow
    p1, status, err = cv2.calcOpticalFlowPyrLK(analyze_frame.old_gray, frame_gray, analyze_frame.p0, None)
    
    # Select good points 
    good_new = p1[status == 1] 
    good_old = analyze_frame.p0[status == 1]
    
    # Preallocate the array to avoid repeated allocation in the loop
    all_movements = np.zeros((len(good_new), 2))
        
    # Using NumPy operations for vectorized computation
    a_b = good_new.reshape(-1, 2)
    c_d = good_old.reshape(-1, 2)

    # Calculate movements
    all_movements = a_b - c_d    
    
    # Calculate the average movement
    delta_movement = (np.mean(all_movements, axis=0).astype(int) * SIZE_FACTOR).astype(int)
    
    analyze_frame.old_gray = frame_gray.copy() 
    analyze_frame.p0 = cv2.goodFeaturesToTrack(analyze_frame.old_gray, mask = None, 
                                    **FEATURE_PARAMS)
    
    if (display_flow := False): 
        black_screen = np.zeros_like(frame)
            
        # Draw lines and circles on the black screen
        for (a, b), (c, d) in zip(a_b, c_d):
            a, b = int(a), int(b)
            c, d = int(c), int(d)
            cv2.line(black_screen, (a, b), (c, d), (255, 255, 255), 2)
            cv2.circle(black_screen, (a, b), 2, 155, -1)
            
        cv2.imshow("Optical flow", black_screen)

    #########################
    ## end of optical flow ##
    #########################
    
    frame = cv2.resize(frame, (0, 0), fx=SIZE_FACTOR, fy=SIZE_FACTOR)

    # get the excess of green 
    exg = funcs.calcluate_exg(frame)
    
    # threshold the image
    tresh = funcs.threshold(exg)

    # open and close the image
    opened_closed = funcs.open_and_close_image(tresh)

    # color the detected green on the original frame
    frame[opened_closed==255] = (0,255,0)

    # get the active solenoids and the speed the robot should move
    solenoid_active = funcs.activate_solenoids(NUMBER_OF_COLS, ROW_PX_FROM_TOP,
                                                opened_closed, solenoid_active, 
                                                THRESHOLD)
    
    #create mask of the sprayed weed
    sprayed = funcs.get_sprayed_weed(NUMBER_OF_COLS, ROW_PX_FROM_TOP, opened_closed, solenoid_active,
                                     SPRAY_RANGE, delta_movement, SPRAY_INTENSITY, SPRAY_SPACING)

    for i in range(1, 256):
        frame[sprayed == i] = (0, 255-i, i)

    # get the speed of the robot
    speed = funcs.get_speed(solenoid_active, MAX_SPEED, MIN_SPEED)

    # calculate the fps
    end_frame = time.time()
    delta_time_frame = end_frame - start_frame
    fps = 1 / delta_time_frame

    # print the UI on the frame
    result = funcs.printUI(frame, NUMBER_OF_COLS, ROW_PX_FROM_TOP,
                            solenoid_active, fps, speed, current_frame,
                            FONT_SCALE, 1)
    

    return result, speed

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
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * SIZE_FACTOR)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * SIZE_FACTOR)
        video.open_video(width, height, OUTPUT_FPS)
        ret, old_frame = cap.read()
        analyze_frame.old_gray = cv2.cvtColor(old_frame, 
                        cv2.COLOR_BGR2GRAY)
        analyze_frame.old_gray = cv2.rotate(analyze_frame.old_gray, cv2.ROTATE_90_CLOCKWISE)
        analyze_frame.p0 = cv2.goodFeaturesToTrack(analyze_frame.old_gray, mask = None, 
                                    **FEATURE_PARAMS)
        if cap.isOpened():
            result = analyze_frame(cap)

    done = False
    
    while cap.isOpened():
        try:
            result, speed = analyze_frame(cap)
            
            if not result.any():
                done = True
            
            # add the frame to the output video
            video.write_frame(result, OUTPUT_FPS, speed, MAX_SPEED, MIN_SPEED, MAX_FPS, MIN_FPS)
            cv2.imshow("Result", result)
            
        except Exception as e:
            print(e)
            video.close_video()
            break
        
        # press q to close the window
        if cv2.waitKey(1) & 0xFF == ord('q') or done:
            cap.release()
            video.close_video()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()