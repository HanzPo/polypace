import cv2
import mediapipe as mp
import numpy as np
import json
from google.protobuf.json_format import MessageToDict

import time
from datetime import datetime
from scipy.stats import norm

DISPLAY_CAMERA = True

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


def calculate_angle(a,b,c):
    # print("start angle calc")
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    v1 = c-b
    v2 = a-b
    
    dot_product = np.dot(v1, v2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # print("done process")
    return np.degrees(np.arccos(dot_product/(norm_a*norm_b)))


# 2 NUMERICAL ERRORS
# 1. SIGMA TOO LARGE, F(X) FOR EVERY X ROUNDS TO 0
# 2. F(A) TOO LARGE FOR A, AND ROUNDS TO 0
# EITHER WAY ATTEMPTS TO DIVIDE BY 0
def custom_pdf(x, loc, scale):
    
    # print("ret: ", norm.pdf(x, loc=loc, scale=scale)/norm.pdf(x, loc=loc, scale=scale))
    # print("passed x: ", x)
    ret = norm.pdf(x, loc=loc, scale=scale)/norm.pdf(0, loc=loc, scale=scale)
    return ret
    # if(str(ret)=="nan"):
    #     # print("ret 0")
    #     return 0


# start = datetime.now()
FR_MULT = 10
def get_curr_time():
    return (datetime.now()-datetime(1970, 1, 1)).total_seconds()*FR_MULT

# print(get_curr_time())

start = get_curr_time()

def delta_t(t2, t1=start):
    # return (t2-t1).total_seconds()*1000
    return t2-t1
    
# def delta_t(t2):
#     return delta_t(start, t2)

def get_angles(landmarks):
    
    # print("start run")
    hip_l=[landmarks[23].x, landmarks[23].y, landmarks[23].z]
    knee_l=[landmarks[25].x, landmarks[25].y, landmarks[25].z]
    ankle_l=[landmarks[27].x, landmarks[27].y, landmarks[27].z]
    hip_r=[landmarks[22].x, landmarks[22].y, landmarks[22].z]
    knee_r=[landmarks[24].x, landmarks[24].y, landmarks[24].z]
    ankle_r=[landmarks[26].x, landmarks[26].y, landmarks[26].z]
    # print("got coords")
    
    hip_knee_ankle_l_theta = calculate_angle(hip_l, knee_l, ankle_l)
    hip_knee_ankle_r_theta = calculate_angle(hip_r, knee_r, ankle_r)
    # print("angle stuff done")
    # delta_t = datetime.now()-start
    # with open("angles.txt", "a") as file:
    #     # Write the new data as a line
    #     # curr_time = int(datetime.now().timestamp()*1000)
    #     # delta_t = curr_time-start
    #     # elapsed = datetime.now() - start
    #     # delta = elapsed.total_seconds()*1000
    #     delta = delta_t(datetime.now())
    #     # print(delta)
    #     file.write(f"{delta} {hip_knee_ankle_l_theta} {hip_knee_ankle_r_theta}\n")
        
        
    #     # print(type(datetime.now().timestamp()*1000))
    #     # delta_t = int((datetime.now()-start).timestamp()*1000)
    #     # print("delta_t: ", delta_t)
    #     # file.write(f"{delta_t} {hip_knee_ankle_l_theta} {hip_knee_ankle_r_theta}\n")
    # print("done write file")
    
    return delta_t(get_curr_time()), hip_knee_ankle_l_theta, hip_knee_ankle_r_theta

def load_data(landmarks):
    
    # print("start run")
    hip_l=[landmarks[23].x, landmarks[23].y, landmarks[23].z]
    knee_l=[landmarks[25].x, landmarks[25].y, landmarks[25].z]
    ankle_l=[landmarks[27].x, landmarks[27].y, landmarks[27].z]
    hip_r=[landmarks[22].x, landmarks[22].y, landmarks[22].z]
    knee_r=[landmarks[24].x, landmarks[24].y, landmarks[24].z]
    ankle_r=[landmarks[26].x, landmarks[26].y, landmarks[26].z]
    # print("got coords")
    
    hip_knee_ankle_l_theta = calculate_angle(hip_l, knee_l, ankle_l)
    hip_knee_ankle_r_theta = calculate_angle(hip_r, knee_r, ankle_r)
    # print("angle stuff done")
    # delta_t = datetime.now()-start
    # with open("angles.txt", "a") as file:
        # Write the new data as a line
        # curr_time = int(datetime.now().timestamp()*1000)
        # delta_t = curr_time-start
        # elapsed = datetime.now() - start
        # delta = elapsed.total_seconds()*1000
        # delta = delta_t(datetime.now())
        # print("op")
        # curr_time = get_curr_time()
        # print(delta)
        # print("aoeu")
        # file.write(f"aoeuoeua\n")
        # file.write(f"{curr_time} {hip_knee_ankle_l_theta} {hip_knee_ankle_r_theta}\n")
        # print("aoeu2")
        
        
    #     # print(type(datetime.now().timestamp()*1000))
    #     # delta_t = int((datetime.now()-start).timestamp()*1000)
    #     # print("delta_t: ", delta_t)
    #     # file.write(f"{delta_t} {hip_knee_ankle_l_theta} {hip_knee_ankle_r_theta}\n")
    # print("done write file")
    
    return delta_t(get_curr_time()), hip_knee_ankle_l_theta, hip_knee_ankle_r_theta



def read_new_test_data(landmarks):
    
    # print("start run")
    hip_l=[landmarks[23].x, landmarks[23].y, landmarks[23].z]
    knee_l=[landmarks[25].x, landmarks[25].y, landmarks[25].z]
    ankle_l=[landmarks[27].x, landmarks[27].y, landmarks[27].z]
    hip_r=[landmarks[22].x, landmarks[22].y, landmarks[22].z]
    knee_r=[landmarks[24].x, landmarks[24].y, landmarks[24].z]
    ankle_r=[landmarks[26].x, landmarks[26].y, landmarks[26].z]
    # print("got coords")
    
    hip_knee_ankle_l_theta = calculate_angle(hip_l, knee_l, ankle_l)
    hip_knee_ankle_r_theta = calculate_angle(hip_r, knee_r, ankle_r)
    # print("angle stuff done")
    # delta_t = datetime.now()-start
    with open("angles.txt", "a") as file:
        # Write the new data as a line
        # curr_time = int(datetime.now().timestamp()*1000)
        # delta_t = curr_time-start
        # elapsed = datetime.now() - start
        # delta = elapsed.total_seconds()*1000
        # delta = delta_t(datetime.now())
        # print("op")
        curr_time = get_curr_time()
        # print(delta)
        # print("aoeu")
        # file.write(f"aoeuoeua\n")
        file.write(f"{curr_time} {hip_knee_ankle_l_theta} {hip_knee_ankle_r_theta}\n")
        # print("aoeu2")
        
        
    #     # print(type(datetime.now().timestamp()*1000))
    #     # delta_t = int((datetime.now()-start).timestamp()*1000)
    #     # print("delta_t: ", delta_t)
    #     # file.write(f"{delta_t} {hip_knee_ankle_l_theta} {hip_knee_ankle_r_theta}\n")
    # print("done write file")
    
    # return delta_t(datetime.now()), hip_knee_ankle_l_theta, hip_knee_ankle_r_theta

prev_theta_l = None
prev_theta_r = None

memory = []
p1=0
p2=0

# If MEMORY_CAP is big enough,
#   p2 should never catch up to p1
#   p1 will not be allowed to surpass p2
MEMORY_CAP = 200
MU = 0
SIGMA = 5  # careful relationship with scaling of time series shit
STD_DEV_CONST = 2.5
COEF = 1

def populate_memory():
    for i in range(0, MEMORY_CAP):
        memory.append(-1)

populate_memory()

def irrelevant_pt(curr_t, t):
    tot_elapsed_t = delta_t(curr_t, t)
    # T -1 ERROR
    print("irr calc: ", delta_t(t, curr_t)/SIGMA)
    print("t:         ", t)
    print("curr time: ", curr_t)
    print("delta t?: ", delta_t(t, curr_t))
    print("norm val: ", custom_pdf(delta_t(t, curr_t), loc=MU, scale=SIGMA))
    # print("norm val: ", norm.pdf(delta_t(t, get_curr_time()), loc=MU, scale=SIGMA))
    print("bool: ", custom_pdf(delta_t(t, curr_t), loc=MU, scale=SIGMA)<0.05)
    # print("bool: ", norm.pdf(delta_t(t, get_curr_time()), loc=MU, scale=SIGMA)<0.05)
    return custom_pdf(delta_t(t, curr_t), loc=MU, scale=SIGMA)<0.05
    # return norm.pdf(delta_t(t, get_curr_time()), loc=MU, scale=SIGMA)<0.05
    # return tot_elapsed_t>STD_DEV_CONST*SIGMA
    # return True
    
    
tot_add = 0
def add_step_to_mem(t):
    
    global tot_add
    
    global p1, p2
    
    memory[p2]=t
    p2=(p2+1)%MEMORY_CAP
    
    print("ADDED PT", tot_add)
    tot_add+=1
    
def rem_steps_from_mem(curr_t):
    global p1, p2
    
    global tot_add
    if(memory[p1]==-1):
        return
    while(irrelevant_pt(curr_t, memory[p1])):
        
        # Never let p1 surpass p2
        if(p1==p2):
            break
        
        memory[p1]=-1
        
        p1=(p1+1)%MEMORY_CAP
        
        print("REMOVED PT", tot_add)
        
        if(p1==p2):
            break
        
def detect_sign_change(t, theta_l, theta_r):
    global prev_theta_l, prev_theta_r
    
    if prev_theta_l==None and prev_theta_r==None:
        prev_theta_l=theta_l
        prev_theta_r=theta_r
        return
    
    # Sign change
    if((theta_r-theta_l)*(prev_theta_r-prev_theta_l)<0):
        print("SIGN CHANGE at ", t)
        
        add_step_to_mem(t)
    
    prev_theta_l=theta_l
    prev_theta_r=theta_r

        
# Called at each iteration
def calculate_speed(landmarks):
    
    global p1, p2
    
    t, theta_l, theta_r = get_angles(landmarks)
    
    detect_sign_change(t, theta_l, theta_r)
    rem_steps_from_mem()
    
    i=p1
    f=0
    while(True):
        
        if(i==p2):
            break
        # f+=1
        f+=norm.pdf(delta_t(memory[i], get_curr_time()), loc=MU, scale=SIGMA)
        i=(i+1)%MEMORY_CAP
    
    return f

def calculate_speed_test(curr_t, t, theta_l, theta_r):
    
    global p1, p2
    
    detect_sign_change(t, theta_l, theta_r)
    rem_steps_from_mem(curr_t)
    
    # print("done p1")
    print(f"p1: {p1}, p2: {p2}")
    out = ""
    print(f"mem[p1]: {memory[p1]}, t: {t}, diff: {delta_t(t, memory[p1])}")
    # print(f"mem[p1]: {memory[p1]}, mem[p2]: {memory[p2]}")
    # for i in range(1, MEMORY_CAP):
    #     out+=f" {memory[i]-memory[i-1]}"
    # print(out)
    # out2 = ""
    # for i in range(MEMORY_CAP):
    #     out2+=f" {memory[i]}"
    # print(out2)
    i=p1
    f=0
    while(1):
        # print("it")
        if(i==p2):
            break
        
        if(memory[i]==-1):
            break
        # print("delt: ", delta_t(memory[i], get_curr_time()))
        # print("mem", memory[i])
        # print("get_curr_time", get_curr_time())
        # print("norm:", norm.pdf(delta_t(memory[i], get_curr_time()), loc=MU, scale=SIGMA))
        # print("pass into norm: ", delta_t(memory[i], get_curr_time())/SIGMA)
        # f+=COEF*norm.pdf(0, loc=MU, scale=SIGMA) # may need
        f+=COEF*custom_pdf(0, loc=MU, scale=SIGMA)
        # f+=COEF*norm.pdf(delta_t(memory[i], get_curr_time()), loc=MU, scale=SIGMA)
        i=(i+1)%MEMORY_CAP
    print("f: ", f)
    return f
    
    
def testing_speed():
    # Open the file in read mode
    
    start_time_sim = None
    with open("rachel_new_data_w_diff_top_only.txt", "r") as file:
        # Iterate through each line in the file
        for line in file:
            # Strip whitespace and split the line by spaces (or other delimiter if applicable)
            elements = line.strip().split()
            
            # Extract the first, second, and third elements as floats
            if len(elements) >= 3:  # Ensure there are at least 3 elements in the line
                first = float(elements[0])
                if(start_time_sim==None):
                    start_time_sim=first
                second = float(elements[1])
                third = float(elements[2])
                
                # # Print the extracted values (optional)
                # print(f"First: {first}, Second: {second}, Third: {third}")
                
                speed = calculate_speed_test(start_time_sim, first/100, second, third)  # careful
                print(speed)
                
                with open("to_plot.txt", "a") as plot_file:
                    plot_file.write(f"{first} {second} {third} {speed}\n")
            else:
                print("Line does not have enough elements:", line)

    print("yurr")

# testing_speed()





mp_drawing  = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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
            # print("haoeu")
            load_data(landmarks)

            # with open("data.json", "w") as json_file:
            #     all_landmarks_dict = {}
            #     for i in range(len(landmarks)):
            #         all_landmarks_dict[NUM_TO_LANDMARK[i]] = MessageToDict(landmarks[i])
                
            #     speed = calculate_speed(landmarks)
                
            #     json.dump({"skeleton": all_landmarks_dict, "speed": speed}, json_file, indent=4)
        
        
        
            landmarks = results.pose_landmarks.landmark
            # load_data(landmarks)

            with open("data.json", "w") as json_file:
                all_landmarks_dict = {}
                for i in range(len(landmarks)):
                    all_landmarks_dict[NUM_TO_LANDMARK[i]] = MessageToDict(landmarks[i])
                
                # speed = calculate_speed(landmarks)
                
                json.dump({"skeleton": all_landmarks_dict}, json_file, indent=4)
                # json.dump({"skeleton": all_landmarks_dict, "speed": speed}, json_file, indent=4)
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