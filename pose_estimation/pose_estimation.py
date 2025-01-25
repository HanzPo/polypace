DISPLAY_CAMERA = True

import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import timedelta
import socket
from datetime import datetime
import json
from google.protobuf.json_format import MessageToDict

NUM_TO_LANDMARK = {
    0: "NOSE",
    1: "LEFT_EYE_INNER",
    2: "LEFT_EYE",
    3: "LEFT_EYE_OUTER",
    4: "RIGHT_EYE_INNER",
    5: "RIGHT_EYE",
    6: "RIGHT_EYE_OUTER",
    7: "LEFT_EAR",
    8: "RIGHT_EAR",
    9: "MOUTH_LEFT",
    10: "MOUTH_RIGHT",
    11: "LEFT_SHOULDER",
    12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW",
    14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",
    16: "RIGHT_WRIST",
    17: "LEFT_PINKY",
    18: "RIGHT_PINKY",
    19: "LEFT_INDEX",
    20: "RIGHT_INDEX",
    21: "LEFT_THUMB",
    22: "RIGHT_THUMB",
    23: "LEFT_HIP",
    24: "RIGHT_HIP",
    25: "LEFT_KNEE",
    26: "RIGHT_KNEE",
    27: "LEFT_ANKLE",
    28: "RIGHT_ANKLE",
    29: "LEFT_HEEL",
    30: "RIGHT_HEEL",
    31: "LEFT_FOOT_INDEX",
    32: "RIGHT_FOOT_INDEX"
}

JSON_NAME = "skeleton"


mp_drawing  = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Lower and upper boundaries for camera
webcam_lower_bound = -0.5
webcam_upper_bound = 1.5

# Calculates the angle between coordinates a, b, and c (which are all x, y coordinates)
def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0])-np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle>180.0:
        angle=360-angle
    
    return angle

# Validates that the x and y value for each land mark (in an array of land marks) is between a lower and upper bound range
def check_validity(land_marks):
    
    for i in land_marks:
        if(i[0]<webcam_lower_bound or i[0]>webcam_upper_bound or i[1]<webcam_lower_bound or i[1]>webcam_upper_bound):
            return False
        
    return True

# Captures webcam footage
cap = cv2.VideoCapture(0)

# Gets tracker
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
    
    # While webcam is open
    while cap.isOpened():

        image = cap.read()[1] # Reads images from video

        # Recolor image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        image.flags.writeable = True  # Make image writable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract Landmarks
        try:
            
            landmarks = results.pose_landmarks.landmark

            with open("coordinates.json", "w") as json_file:
                all_landmarks_dict = {}
                for i in range(len(landmarks)):
                    all_landmarks_dict[NUM_TO_LANDMARK[i]] = MessageToDict(landmarks[i])
                
                json.dump({JSON_NAME: all_landmarks_dict}, json_file, indent=4)
        except:
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )
        
        if(DISPLAY_CAMERA):
            cv2.imshow('Mediapipe Feed', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
    cap.release()