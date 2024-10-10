from ultralytics import YOLO
import cv2
import cvzone
import math

cap=cv2.VideoCapture("ppe1.mp4")
model=YOLO("ppe.pt")
className=['Hardhat','Mask','NO-Hardhat','NO-Mask','NO-Safety Vest','Person','Safety Cone','Safety Vest','Machinery','vehicle']
mycolor=(0,0,255)
while True:
    success,img=cap.read()
    results=model(img,stream=True)
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1

            conf=math.ceil((box.conf[0]*100))/100
            cls=int(box.cls[0])
            currentClass=className[cls]
            if conf > 0.5:
                if currentClass=='NO-Harthat' or currentClass=='NO-Safeti Vest' or currentClass=='No-Mask':
                    mycolor=(0,0,255)
                if currentClass=='Harthat' or currentClass=='Safeti Vest' or currentClass == 'Mask':
                    mycolor=(0,255,0)
                else:
                    mycolor=(255,0,0)
                cvzone.putTextRect(img,f'{className[cls]} {conf}',(max(0,x1),max(35,y1)),scale=1,thickness=1,colorB=mycolor,colorT=(255,255,255),colorR=mycolor,offset=5)
                cv2.rectangle(img,(x1,y1),(x2,y2),mycolor,3)
    
    cv2.imshow("image",img)
    cv2.waitKey(1)