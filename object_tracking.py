import cv2
import os
import numpy as np
from object_detection import ObjectDetection
import math
from datetime import datetime
from playsound import playsound

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture(0)

classes = od.load_class_names()
print(classes)

now = datetime.now()

while True:
    print("New frame")
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)
    
    num = len(class_ids)
    #print(od.detect(frame))
    
    for i in range(0,num):
        (x, y, w, h) = boxes[i]
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)

        print("FRAME N°", " ", x, y, w, h)
        
        label = str(classes[class_ids[i]])
        print(label)
        print(class_ids[i])
        print(scores[i])
        color = (255, 0, 0)
        
        if label == "person":
            color = (0, 0, 255)
            # save object detected
            cv2.imwrite('Output/image'+ now.strftime("%d%m%Y_%H%M%S") +'.jpg', frame)
            # ring alarm
            playsound('dnn_model/Alarm01.wav')
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y), 0, 1, color, 3)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 2:
        break

cap.release()
cv2.destroyAllWindows()
