import os
import cv2
import math
import shutil
import pytesseract
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


#Module to detect whether the object in the video sequence is moving or not.
def MovementDetection(VideoPath):

    if(os.path.exists(VideoPath) == False):
        print("Error(In MovementDetection): Path does not exists")
        return []

    cap =cv2.VideoCapture(VideoPath)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    image_list=[] #For storing image frames
    while(cap.isOpened()):

        original_frame1=frame1.copy()

        #Computes absoulute difference between two consecutive frames
        diff = cv2.absdiff(frame2, frame1)
        #Converts the frame obtained to grayscale
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        #Preprocessing of the frame
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _,thresh=cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)

        contours, _=cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        frame1=cv2.line(frame1, (0, 290), (640, 290), (0,255,0), 2)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if(cv2.contourArea(contour) < 800):
                continue
            else:
                #Region of Interest(ROI)
                if((y>=250) and (y<= 275)):
                    image_list.append(original_frame1)

        frame1=frame2
        ret, frame2 = cap.read()
        if not ret:
            break;
        ret, frame=cap.read()

    cap.release()
    return image_list

# Vehicle Detection
def DetectVehicle(img, net, classes):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    #img = cv2.imread(image)
    img = cv2.resize(img, None, fx=1, fy=1)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    flag = False
    x,y,w,h=0,0,0,0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "car":
                flag = True
                break

    return flag,(x,y,w,h)


def Validation_And_RatioTest(objectArea):
    (x,y),(width,height),angle=objectArea

    aspectRatio=0
    if width>height:
            aspectRatio=width/height
    else:
        aspectRatio=height/width
    if aspectRatio<3 or aspectRatio>10:
        return False

    return True

def width_greater_(contour):

    maxLength=contour[0][0][0]
    minLength=contour[0][0][0]

    maxWidth=contour[0][0][1]
    minWidth=contour[0][0][1]

    for i in contour:
        if maxLength<i[0][0]:
            maxLength=i[0][0]

        elif minLength>i[0][0]:
            minLength=i[0][0]

        if maxWidth<i[0][1]:
            maxWidth=i[0][1]

        elif minWidth>i[0][1]:
            maxWidth=i[0][1]

    length=maxLength-minLength
    width=maxWidth-minWidth
    if length < width or length<100 or width>50 or length>250:
        return True


    return False;



# Returning license_plate from this function
def plate_detection(img):

    #blurring to remove high frequency component
    imgBlurred = cv2.GaussianBlur(img, (5,5), 0)

    #color to greyscale image
    gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)

    #otsu thresolding
    ret2,threshold_img = cv2.threshold(blackhat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    morph_img_threshold = threshold_img.copy()

    # rectangular kernel for morphological operation
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(20, 1))  #(20,10)

    #morphological operation(dilation)
    cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)

    #added new
    thresh = cv2.erode(morph_img_threshold, None, iterations=2)

    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, hierarchy= cv2.findContours(morph_img_threshold,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)

    #2 with Validation
    # i is count index, cnt is coordinates of 1 whole object
    for i,cnt  in enumerate(contours):
        #it is observed that for  most of image , number of coordinates number plate region lies between 300-500
        if(len(cnt)>200):
            if width_greater_(cnt):
                continue;

            objectArea=cv2.minAreaRect(cnt)
            if Validation_And_RatioTest(objectArea):
                x, y, w, h = cv2.boundingRect(cnt)
                license_plate = img[y:y + h + 5, x-5:x + w]
                #cv2.imwrite("plate.png", license_plate)
                return license_plate

# Functions needed to be loaded only once for many all the recognitions..
'''
Function used to rotate the image by angle along the center using warp affine
'''
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


'''
Sorting the contours from top to bottom, left to right using bounding boxes
'''
def sort_contours(cnts,reverse = False):
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    boundingBoxes = sorted(boundingBoxes, key=lambda b:b[0]+b[1], reverse=False)
    return boundingBoxes

'''
Predicting label for image size 128*128
'''
def predict_from_model_128(image,model,labels):
    image = cv2.resize(image,(128,128))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction


'''
Predicting label for image size 80*80
'''
def predict_from_model_80(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction

'''
Using outlier detection technique to remove all the contours except the character contours.
'''
def minimum_character(input_list):

    if len(input_list) == 0:
        return -1
    A = np.array(sorted(input_list))
    #print(A)
    z = np.abs(stats.zscore(A))
    A_clean = np.array([A[i] for i in range(len(z)) if z[i] < 2])
    return A_clean[0]



'''
recognize_char function is combination of hough transform alogrithm, character segmentation, character recognition.
'''
def recognize_char(input_img, first_letter_model,first_letter_labels, second_letter_model, second_letter_labels,digit_model, digit_labels, model, labels):


    '''
    Pre-processing the image to maintain the edges of the characters while supressing the noise in the image.
    Obtaining the canny edge image for the gray scaled image.
    '''
    blur = cv2.bilateralFilter(input_img,9,95,95)
    blur = cv2.detailEnhance(blur, 5, 0.95)
    gray_img = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    canny_edges = cv2.Canny(gray_img,220, 250, None, 5)

    '''
    Using hough transform over the image to get all the lines between +- 15 degrees of the lines to obtain the skew in the image.
    '''

    cv2.imwrite("PlateImg.png", input_img)
    lines = cv2.HoughLines(canny_edges, 1, np.pi / 180, 50, None, 0, 0)
    avg_theta = 0
    theta =0
    line_count = 0
    if lines is not None:
        for i in range(0, len(lines)):
            theta = lines[i][0][1]
            angle = (180*theta/3.1415926 - 90)
            if -15<=angle<=15:
                avg_theta += angle
                line_count += 1
    if line_count != 0:
        avg_theta = avg_theta/line_count
        img_rotated = rotate_image(input_img, avg_theta)
    else:
        img_rotated = rotate_image(input_img, 0)

    '''
    Pre-processing the rotated image as before.
    '''

    enhanced_img = cv2.detailEnhance(img_rotated, 9, 10, 0.5)
    enhanced_gray_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    blur_enhanced_gray_img = cv2.bilateralFilter(enhanced_gray_img, 9,10,10)
    binary_image = cv2.threshold(blur_enhanced_gray_img, 100,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    '''
    Getting the external hierarchy contours on the image.
    '''
    cont, _  = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_segmented = []
    char_w = 96
    char_h = 96

    char_ratio_list = []
    sorted_bounding_boxes = sort_contours(cont)


    '''
    Using the ratio of h/w for removal of outlier contours.
    '''
    for box in sorted_bounding_boxes:
        (x, y, w, h) = box
        ratio = h/w
        if 1<=ratio<=5:
            char_ratio_list.append(h/img_rotated.shape[0])

    mini_char = minimum_character(char_ratio_list)
    print(mini_char)
    if mini_char == -1:
        flag = False
    else:
        flag = True


    if(flag):
        '''
            Segmenting each character..
        '''
        for box in sorted_bounding_boxes:
            (x, y, w, h) = box
            ratio = h/w
            if 1<=ratio<=5:
                if 0.3 <= h/img_rotated.shape[0] <= 0.9:
                    curr_num = binary_image[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(char_w, char_h))
                    _, curr_num = cv2.threshold(curr_num, 90, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    char_segmented.append(curr_num)

        if len(char_segmented) < 5:
            return "ERROR_CHAR_LEN"


        '''
        Using the indian number plate constraints to recognize the characters and generating the final string..
        '''
        final_string = ''
        title = np.array2string(predict_from_model_128(char_segmented[0],first_letter_model,first_letter_labels))
        final_string +=title.strip("'[]")
        title = np.array2string(predict_from_model_128(char_segmented[1],second_letter_model,second_letter_labels))
        final_string +=title.strip("'[]")

        title = np.array2string(predict_from_model_128(char_segmented[2],digit_model,digit_labels))
        final_string +=title.strip("'[]")
        title = np.array2string(predict_from_model_128(char_segmented[3],digit_model,digit_labels))
        final_string +=title.strip("'[]")

        for i in range(4,len(char_segmented)-4):
            title = np.array2string(predict_from_model_80(char_segmented[i],model,labels))
            final_string+=title.strip("'[]")

        for i in range(len(char_segmented)-4,len(char_segmented)):
            title = np.array2string(predict_from_model_128(char_segmented[i],digit_model,digit_labels))
            final_string+=title.strip("'[]")

        return final_string
    else:
        return pytesseract.image_to_string(blur_enhanced_gray_img)
