import cv2
from pathlib import Path
import numpy as np

def prptrgdata(pt):
    img_names=Path(pt).glob("*.*")
    faces=[]
    for img_name in img_names:
        img=cv2.imread(pt+"/"+img_name.name)
        face=detectface(img)
        if face is not None:
            faces.append(face)
    return faces

def detectface(img):
    gryimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=fc.detectMultiScale(gryimg,1.2,4)
    if len(faces)==0:
        return None
    else:
        (x,y,w,h)=faces[0]
        return gryimg[y:y+h,x:x+w]

def predict(img_path):
    img=cv2.imread(img_path)
    face=detectface(img)
    lbl,cfn=fc_rec.predict(face)
    print(cfn)
    if cfn>=0 and cfn<=100:
        nam=str(lbl)+".jpg"
        fndimg=cv2.imread("images/"+nam)
        gryimg=cv2.cvtColor(fndimg,cv2.COLOR_BGR2GRAY)
        fcfnd=fc.detectMultiScale(gryimg,1.3,4)
        (x,y,w,h)=fcfnd[0]
        cv2.rectangle(fndimg,(x,y),(x+w,y+h),(0,255,0))
        cv2.putText(fndimg,"Found",(x,y),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255))
        cv2.imshow("Matching Result",fndimg)
        cv2.waitKey(0)
    else:
        print("Image not found")



#fc=cv2.CascadeClassifier("utils/haarcascade_frontalface_default.xml")
#fc_rec=cv2.face.LBPHFaceRecognizer_create()
#faces=prptrgdata("images")
#fc_rec.train(faces,np.arange(1,len(faces)+1))
#predict("img/3.jpg")

img=cv2.imread("img/1.jpg")
gryimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
fc=cv2.CascadeClassifier("utils/haarcascade_frontalface_default.xml")
faces=fc.detectMultiScale(gryimg,1.5,5)
for (x,y,w,h) in faces:
 cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow("My Album",img)
cv2.waitKey(0)


