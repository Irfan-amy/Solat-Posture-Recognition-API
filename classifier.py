from io import BytesIO

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image


model = None

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End


    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
        
    return np.degrees(angle) 



def preprocess(image):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode= False, model_complexity=1,min_detection_confidence=0.5) as pose:
        frame = image
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        image_height, image_width, _ = image.shape
        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )   
        data = []
        try:
            landmarks = results.pose_landmarks.landmark
            
            

            if (((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > 0.75 and
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility > 0.75 and
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility > 0.75) or 
            (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > 0.75 and
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility > 0.75 and
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility > 0.75)) and
            ((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > 0.75 and
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility > 0.75 and
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > 0.75) or 
            (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > 0.75 and
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility > 0.75 and
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > 0.75)) and 
            ((landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility > 0.75 and
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > 0.75 and
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility > 0.75) or 
            (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility > 0.75 and
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > 0.75 and
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility > 0.75))):
                
            
                
                

                if (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > 0.75 and
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility > 0.75 and
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility > 0.75) :
                    angle = calculate_angle(
                        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image_height,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z],
                        [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image_height,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z],
                        [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image_height,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z],
                    )

                    data.append(angle)
                elif (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > 0.75 and
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility > 0.75 and
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility > 0.75) :
                    angle = calculate_angle(
                        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image_height,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z],
                        [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image_height,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z],
                        [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image_height,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z],
                    )
                    data.append(angle)
                else: 
                    print("Error")

                if (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > 0.75 and
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility > 0.75 and
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > 0.75) :
                    angle = calculate_angle(
                        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image_height,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z],
                        [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * image_height,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z],
                        [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * image_height,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z],
                    )
                    data.append(angle)
                elif (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > 0.75 and
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility > 0.75 and
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > 0.75) :
                    angle = calculate_angle(
                        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image_height,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z],
                        [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * image_height,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z],
                        [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * image_height,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z],
                    )
                    data.append(angle)
                else: 
                    print("Error")

                if (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility > 0.75 and
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > 0.75 and
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility > 0.75) :
                    angle = calculate_angle(
                        [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * image_height,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z],
                        [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * image_height,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z],
                        [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * image_height,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z],
                    )
                    data.append(angle)
                elif (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility > 0.75 and
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > 0.75 and
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility > 0.75) :
                    angle = calculate_angle(
                        [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * image_height,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z],
                        [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * image_height,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z],
                        [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * image_width,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * image_height,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z],
                    )
                    data.append(angle)
                else: 
                    print("Error")
                
                
                return data
            
                        
        except:
            return "Misiing"

def predict(image):
    global model
    if model is None:
        model = load_model("model.h5")
    
    
    labels = ["Sujud","Jalsa","Ruku"]
    angles = preprocess(image)

    result = model.predict(np.asarray(angles).reshape((1,3)))
    result = np.argmax(result,axis=1)
    # response = []
    # for i, res in enumerate(result):
    #     resp = {}
    #     resp["class"] = res[1]
    #     resp["confidence"] = f"{res[2]*100:0.2f} %"

    #     response.append(resp)

    return labels[result[0]]


def read_imagefile(file):
    pil_image = Image.open(BytesIO(file)).convert('RGB')
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    return open_cv_image