import cv2
import dlib
import numpy as np
import os

class MakeData:
    
    def __init__(self, ):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        self.left_eye = [36, 37, 38, 39, 40, 41]
        self.right_eye = [42, 43, 44, 45, 46, 47]
    
    def setThreshold(self, threshold):
        pass

    def featureLocs(self, features):
        locs = [(features.part(i).x, features.part(i).y) for i in range(68)]
        return locs
    
    def eyeMask(self, frame, features, left=True):
        mask = np.zeros(frame.shape, dtype=np.uint8)
        if left:
            eye_markers = self.left_eye
        else:
            eye_markers = self.right_eye

        eye_points = [features[i] for i in eye_markers]
        mask = cv2.fillConvexPoly(mask, np.array(eye_points,dtype=np.int32), 255)
        return mask

    def getEyes(self, frame, gray_frame, box):
        features = self.predictor(gray_frame, box)
        self.flocs = self.featureLocs(features)
        left_eye_mask = self.eyeMask(gray_frame, self.flocs, left=True)
        right_eye_mask = self.eyeMask(gray_frame, self.flocs, left=False)
        eye_mask = np.logical_or(left_eye_mask, right_eye_mask)
        eye_mask = np.uint8(eye_mask)
        eye_mask = cv2.dilate(eye_mask, np.ones((9,9), dtype=np.uint8), 5)
        eyes = cv2.bitwise_and(frame, frame, mask=eye_mask)
        mask = (eyes == [0,0,0]).all(2)
        eyes[mask] = [255,255,255]
        return eyes
    
    def morph(self, roi):
        threshold = cv2.getTrackbarPos('threshold', 'threshold')
        _, eyes_binary = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)
        eyes_eroded = cv2.erode(eyes_binary, None, iterations=2)
        eyes_dilated = cv2.dilate(eyes_eroded, None, iterations=4)
        eyes_smoothed = cv2.medianBlur(eyes_dilated, 3)
        eye_blobs = cv2.bitwise_not(eyes_smoothed)

        return eye_blobs
    
    def contourSegmentation(self, frame, blobs, bridge, left=True):
        centerx, centery = -1000, -1000
        contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        try:
            pupil = max(contours, key = cv2.contourArea)
            moments = cv2.moments(pupil)
            # print(moments)
            centerx = int(moments['m10']//(moments['m00']+1e-5))
            centery = int(moments['m01']//(moments['m00']+1e-5))
            # print(centerx, centery)
            if not left:
                centerx += bridge
            cv2.rectangle(frame, (centerx-3, centery-3), (centerx+3, centery+3), (0,0,255), 2)
            cv2.imshow('eyes', frame)
            cv2.imshow("image", blobs)
            cv2.waitKey(100)
        except Exception as e:
            print(e)
            pass
        return centerx, centery


    def test(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.detector(gray_frame, 1)
        if len(boxes):
            for box in boxes:
                eyes = self.getEyes(frame, gray_frame, box)
                eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
                blobs = self.morph(eyes_gray)
                # cv2.imshow("blobs",blobs)
                # cv2.waitKey(100)
                bridge = (self.flocs[42][0] + self.flocs[39][0])//2
                leftx, lefty = self.contourSegmentation(frame, blobs[:,:bridge], bridge, left=True)
                rightx, righty = self.contourSegmentation(frame, blobs[:,bridge:], bridge, left=False)
            

            
        

if __name__ == "__main__":
    md = MakeData()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('threshold')
    cv2.createTrackbar('threshold', 'threshold', 0, 255, md.setThreshold)
    _, threshold_image = cap.read()
    while 1:
        _, frame = cap.read()
        md.test(frame)
    path = "C:/Users/prane/Downloads/eye_data/images/train/"
    # folders = os.listdir(path)
    # folders=folders[:-3]
    # images = os.listdir(path)
    # img = cv2.imread(path+images[10])
    # cv2.imshow("read", img)
    # md.test(img)
    # print(len(images))
    # for x in folders:
    #     images = os.listdir(path+x+'/frames')
    #     print(len(images))
    cap.release()
    cv2.destroyAllWindows()

        
