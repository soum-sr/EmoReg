import cv2
import numpy as np 
from keras.models import load_model

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.model = load_model('model.h5')
        self.emotion_dict = {0:"Angry", 1:"Disgust", 2: "Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        frame = cv2.flip(frame, 1)

        # Convert to bnw
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x-25, y-25), (x+w+25, y+h+25) , (0,255,0), 1)
            roi_gray = gray[y:y+h, x:x+w+25]

            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48,48)), -1),0)
            cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            prediction = self.model.predict(cropped_img)
            cv2.putText(frame, self.emotion_dict[int(np.argmax(prediction))], (x,y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 1, cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
