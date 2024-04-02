import dlib
import cv2
import numpy as np
import os

class PupilTracker:
    
    def __init__(self, display=0):
        # HOG + Linear SVM face detection
        self.detector = dlib.get_frontal_face_detector()
        
        # Facial landmark detection
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # landmark identities correponding to eyes
        self.left_eye = [36, 37, 38, 39, 40, 41]
        self.right_eye = [42, 43, 44, 45, 46, 47]

        # size of the pupil wrt the cornea
        self.average_pupil_ratio=0.48

        #enable/display video output
        self.display=display

    
    def featureLocs(self, features):
        '''
        fetches the image coordinates of the eye landmarks
        
        inputs
        features - dlib point object
        
        outputs
        image coordinated for eye landmarks.
        '''
        
        locs = [(features.part(i).x, features.part(i).y) for i in range(68)]
        return locs
    
    def eyeMask(self, frame, features, left=True):
        '''
        takes an image and the image coordinates for an eye as input and returns a masked image with the background separarted from the object(eye)
        
        inputs
        frame - image
        features - eye coordinates in the image
        left - left eye or right eye
        
        outputs
        masked image of the eye
        '''
        
        mask = np.zeros(frame.shape, dtype=np.uint8)
        if left:
            eye_markers = self.left_eye
        else:
            eye_markers = self.right_eye

        eye_points = [features[i] for i in eye_markers]
        mask = cv2.fillConvexPoly(mask, np.array(eye_points,dtype=np.int32), 255)
        return mask

    def getEyes(self, frame, gray_frame, box):
        '''
        call the functions to extract the eye masks and combine the masks for left and right eyes.

        Inputs
        frame - image
        gray_frame - gray image
        box - facial landmarks

        Outputs
        processed left eye, right eye and both eyes masks
        '''

        features = self.predictor(gray_frame, box)
        self.flocs = self.featureLocs(features)
        left_eye_mask = self.eyeMask(gray_frame, self.flocs, left=True)
        right_eye_mask = self.eyeMask(gray_frame, self.flocs, left=False)
        eye_mask = np.logical_or(left_eye_mask, right_eye_mask)
        eye_mask = np.uint8(eye_mask)
        left_eye = cv2.bitwise_and(gray_frame, gray_frame, mask=np.uint8(left_eye_mask))
        right_eye = cv2.bitwise_and(gray_frame, gray_frame, mask=np.uint8(right_eye_mask))
        # eye_mask = cv2.dilate(eye_mask, np.ones((9,9), dtype=np.uint8), 5)
        eyes = cv2.bitwise_and(frame, frame, mask=eye_mask)
        mask = (eyes == [0,0,0]).all(2)
        eyes[mask] = [255,255,255]
        return eyes, left_eye, right_eye
    
    def morph(self, roi, threshold):
        '''
        Morphological operations to enhance the binary image and making the image uniform by removing any specular noise.

        Inputs
        roi - image with eyes separated from the background.
        threshold - threshold to binarize the image.

        Outputs
        enhanced binary image with eyes shaded white and the background dark.
        '''

        _, eyes_binary = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY)
        if self.display:
            cv2.imshow("",eyes_binary)
            cv2.waitKey(100)
        eyes_eroded = cv2.erode(eyes_binary, None, iterations=2)
        eyes_dilated = cv2.dilate(eyes_eroded, None, iterations=4)
        eyes_smoothed = cv2.medianBlur(eyes_dilated, 3)
        eye_blobs = cv2.bitwise_not(eyes_smoothed)

        return eye_blobs
    
    def contourSegmentation(self, frame, blobs, bridge, left=True):
        '''
        find the pupils by segmenting the eye region, the region with maximum area is treated as a pupil, estimate the pupil centroids.

        inputs
        frame - the original image
        blobs - eyes
        bridge - nose bridge to perform sanity checks of the localized pupils
        left - left/right eye

        outputs
        centroids for left and right eyes
        '''
        
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
        except Exception as e:
            # print(e)
            pass
        return centerx, centery

    def findThreshold(self, eye, left=True):
        '''
        adaptive thresholding finds appropriate threshold for both the eys. Best threshold finds the pupil to cornea ratio close to 0.48, avergage emperical ratio.

        inputs
        eye - the masked eye
        left - left/right eye

        outptut
        best threshold
        '''
        
        if left:
            eye_points = np.array([self.flocs[i] for i in self.left_eye])
        else:
            eye_points = np.array([self.flocs[i] for i in self.right_eye])
        minx = np.min(eye_points[:,0])
        maxx = np.max(eye_points[:,0])
        miny = np.min(eye_points[:,1])
        maxy = np.max(eye_points[:,1])

        if left:
            self.left_origin = (minx, miny)
        else:
            self.right_origin = (minx, miny)

        crop_eye = eye[miny:maxy, minx:maxx]
        kernel = np.ones((3, 3), np.uint8)
        processed_eye = cv2.bilateralFilter(crop_eye, 10, 15, 15)
        processed_eye = cv2.erode(crop_eye, kernel, iterations=3)
        
        calib={}
        for t in range(10, 250, 10):
            _, thresh_eye = cv2.threshold(processed_eye, t, 255, cv2.THRESH_BINARY)
            
            calib[t] = self.pupilRatio(thresh_eye)
        # print(calib)
        best_threshold, iris_size = min(calib.items(), key=(lambda p: abs(p[1] - self.average_pupil_ratio)))
        return best_threshold, cv2.threshold(processed_eye, best_threshold, 255, cv2.THRESH_BINARY)[1]


    def pupilRatio(self, eye):
        '''
        copmute pupil to cornea ratio

        inputs
        eye - masked eye
        '''
        
        h, w = eye.shape
        total_area = h * w
        pupil_area = total_area - cv2.countNonZero(eye)
        return pupil_area / total_area
   
    def test(self, frame):
        '''
        putting everything together

        inputs
        frame - original image from video capture

        outputs
        face coordinates and pupil centroids.
        '''
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(frame, 1)
        rects = {}
        eye_rects = {}
        if not len(faces):
            return None, None
        for i, face in enumerate(faces):
            rects[i] = face
            eyes, leye, reye = self.getEyes(frame, gray_frame, rects[i])
            eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
            blobs = self.morph(eyes_gray, 150)
            bridge = (self.flocs[42][0] + self.flocs[39][0])//2
            leftx, lefty = self.contourSegmentation(frame, blobs[:,:bridge], bridge, left=True)
            rightx, righty = self.contourSegmentation(frame, blobs[:,bridge:], bridge, left=False)
            # if self.display:
            #     cv2.rectangle(frame, (leftx-3, lefty-3), (leftx+3, lefty+3), (0,0,255), 2)
            #     cv2.rectangle(frame, (rightx-3, righty-3), (rightx+3, righty+3), (0,0,255), 2)
            #     cv2.putText(frame, )
            #     cv2.imshow("image", frame)
            #     cv2.waitKey(100)
            # print(leftx, lefty, rightx, righty)
            eye_rects[i] = (leftx, lefty, rightx, righty)
        return eye_rects, rects

if __name__ == "__main__":
    pt = PupilTracker()
    cap = cv2.VideoCapture(0)
    # cv2.namedWindow('threshold')
    # cv2.createTrackbar('threshold', 'threshold', 0, 255, md.setThreshold)
    _, threshold_image = cap.read()
    while 1:
        _, frame = cap.read()
        pt.test(frame)
    cap.release()
    # pupils, _ = cv2.destroyAllWindows()
    # print(pupils)
