import PIL
import cv2 as cv
import numpy as np
import json
import os
import shutil

path = "C:/Users/prane/Downloads/eye_data/"

folders = os.listdir(path)

folders=folders[:-3]

print(folders)

def annotate(img, face, eye, i):
    img_h,img_w = np.asarray(img).shape[:2]

    face_abs_x = face['X'][i]
    face_abs_y = face['Y'][i]
    face_width = face['W'][i]
    face_height = face['H'][i]

    eye_abs_x = eye['X'][i]
    eye_abs_y = eye['Y'][i]
    eye_width = eye['W'][i]
    eye_height = eye['H'][i]

    x = (face_abs_x+eye_abs_x)
    y = (face_abs_y+eye_abs_y)
    w = eye_width
    h = eye_height

    cx = (x+w/2)/img_w
    cy = (y+h/2)/img_h
    rw = w/img_w
    rh = h/img_h

    return cx, cy, rw, rh

for x in folders:
    images = os.listdir(path+x+'/frames')
    print(len(images))
    num_train = np.ceil(0.8*len(images))
    num_val = np.ceil(0.1*len(images))

    
    with open(path+x+'/appleFace.json') as f:
        faces = json.load(f)

    with open(path+x+'/appleLeftEye.json') as f:
        left_eyes = json.load(f)

    with open(path+x+'/appleRightEye.json') as f:
        right_eyes = json.load(f)

    for i in range(len(left_eyes['IsValid'])):
        try:
            if (left_eyes['IsValid'][i] or right_eyes['IsValid'][i]) and i < num_train:
                # print("train")
                # print(i)
                # print(x)
                # print(images[i])
                shutil.copy(path+x+'/frames/'+images[i], path+'train/'+x+images[i])
                img = cv.imread(path+'train/'+x+images[i])
                f = open(path+'train/'+x+images[i].strip(".jpg")+'.txt', 'w')
                if left_eyes['IsValid'][i]:
                    cx,cy,w,h = annotate(img,faces,left_eyes,i)
                    f.write("0 "+str(cx)+" "+str(cy)+" "+str(w)+" "+str(h)+"\n")
                if right_eyes['IsValid'][i]:
                    cx,cy,w,h = annotate(img,faces,right_eyes,i)
                    f.write("0 "+str(cx)+" "+str(cy)+" "+str(w)+" "+str(h))
                f.close()

            if i >= num_train and i < (num_train+num_val) and (left_eyes['IsValid'][i] or right_eyes['IsValid'][i]):
                # print("val")
                # print(i)
                shutil.copy(path+x+'/frames/'+images[i], path+'validation/'+x+images[i])
                img = cv.imread(path+'validation/'+x+images[i])
                f = open(path+'validation/'+x+images[i].strip(".jpg")+'.txt', 'w')
                if left_eyes['IsValid'][i]:
                    cx,cy,w,h = annotate(img,faces,left_eyes,i)
                    f.write("0 "+str(cx)+" "+str(cy)+" "+str(w)+" "+str(h)+"\n")
                if right_eyes['IsValid'][i]:
                    cx,cy,w,h = annotate(img,faces,right_eyes,i)
                    f.write("0 "+str(cx)+" "+str(cy)+" "+str(w)+" "+str(h))
                f.close()
            
            if i >= num_train+num_val and i < len(left_eyes['IsValid']) and (left_eyes['IsValid'][i] or right_eyes['IsValid'][i]):
                # print("test")
                # print(i)
                shutil.copy(path+x+'/frames/'+images[i], path+'test/'+x+images[i])
                img = cv.imread(path+'test/'+x+images[i])
                f = open(path+'test/'+x+images[i].strip(".jpg")+'.txt', 'w')
                if left_eyes['IsValid'][i]:
                    cx,cy,w,h = annotate(img,faces,left_eyes,i)
                    f.write("0 "+str(cx)+" "+str(cy)+" "+str(w)+" "+str(h)+"\n")
                if right_eyes['IsValid'][i]:
                    cx,cy,w,h = annotate(img,faces,right_eyes,i)
                    f.write("0 "+str(cx)+" "+str(cy)+" "+str(w)+" "+str(h))
                f.close()
        except:
            print(i)
        


        
