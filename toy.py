# https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
# https://developers.google.com/mediapipe/api/solutions
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=_JVO3rvPD4RN
# https://medium.com/@oetalmage16/a-tutorial-on-finger-counting-in-real-time-video-in-python-with-opencv-and-mediapipe-114a988df46a
# https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb?short_path=4e1f464
# https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/drawing_styles.py
# https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/drawing_utils.py?ref=assemblyai.com



import numpy as np
import time
import cv2 
#import keyboard
from pynput.mouse import Button, Controller
import drawing_styles_custom
from screeninfo import get_monitors

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2


#================================================================================

def driver():
  
    camera = cv2.VideoCapture(0) 
    #camera.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
    #camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
    mouse  = Controller()

    #-----------------------------------------------

    landmarker_result = mp.tasks.vision.HandLandmarkerResult
    def reset_landmarker_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        global landmarker_result
        landmarker_result = result            
    options = mp.tasks.vision.HandLandmarkerOptions(
                 base_options = mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task'),
                 running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM,
                 num_hands=1,
                 min_hand_detection_confidence=0.5,
                 min_hand_presence_confidence=0.5,
                 min_tracking_confidence=0.5,
                 result_callback = reset_landmarker_result
              )
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options) 

    #-----------------------------------------------

    monitor = get_monitors()[0]           # --> monitor.width, monitor.height

    #-----------------------------------------------

    def draw_landmarks_on_image(image):
        global landmarker_result

        try:
            if landmarker_result.hand_landmarks == []:
                return image

            annotated_image = np.copy(image)

            hand_landmarks = landmarker_result.hand_landmarks[0]

            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, 
                                                y=landmark.y, 
                                                z=landmark.z) 
                for landmark in hand_landmarks])

            indexTip_x = hand_landmarks[8].x
            indexTip_y = hand_landmarks[8].y

            img_H, img_W, _ = annotated_image.shape

            if indexTip_x >= 0.4 and indexTip_y*monitor.height >= 70:
                mouse.press(Button.left)
                mouse.position = (int(indexTip_x*monitor.width), int(indexTip_y*monitor.height))
                mouse.release(Button.left)

            one_or_two = 1 if indexTip_x < 0.4 else 2    # color index finger depending on which side of the screen

            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                drawing_styles_custom.get_hand_landmarks_style(one_or_two),
                drawing_styles_custom.get_hand_connections_style(one_or_two)
            )
            return annotated_image
        except:
            return image

    #-----------------------------------------------
      
    while(True): 
       
        returnValue, frame = camera.read()     # capture a frame from camera
        frame = cv2.flip(frame, 1)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, int(time.time()*1000))

        frame = draw_landmarks_on_image(frame)
       
        cv2.imshow('frame', frame)             # show the frame
           
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    #-----------------------------------------------

    camera.release()
    cv2.destroyAllWindows() 

#================================================================================

if __name__ == "__main__": driver()

