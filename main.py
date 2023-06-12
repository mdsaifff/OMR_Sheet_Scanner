
import cv2
import numpy as np
import imutils
from imutils import contours
from imutils.perspective import four_point_transform

answer_key={0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

path='omr_result_02.png'


image=cv2.imread(path)
gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred= cv2.GaussianBlur(gray, (5,5),0)
edged= cv2.Canny(blurred, 75,200)
#cv2.imshow("Edged Image", edged)

#find document paper
cnts= cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts= imutils.grab_contours(cnts)
doc_cnt= None
#cv2.drawContours(image,cnts,-1,(0,255,0),2) #draw contours
#cv2.imshow('Contours in Green',image) #show contours in green


if len(cnts)>0:
    cnts=sorted(cnts, key=cv2.contourArea, reverse= True)

    for c in cnts:                   # approximate the contour
        peri=cv2.arcLength(c, True)
        approx= cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:        #if approximated contour has 4 side, then paper is found
            doc_cnt = approx
            break

#BIRD'S VIEW
paper = four_point_transform(image, doc_cnt.reshape(4, 2))
warped = four_point_transform(gray, doc_cnt.reshape(4, 2))



#threshold (binarize)
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


#finding marked bubble
#cv2.imshow(' Threshold Image', thresh)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

for c in cnts: 		#create bounding box of the contour and use it find aspect ratio
	(x, y, w, h) = cv2.boundingRect(c)             #rectangle
	ar = w / float(h)
	#have an aspect ratio approximately equal to 1 and at least 20 pixels in both dimensions
	if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
		questionCnts.append(c)

questionCnts=contours.sort_contours(questionCnts, method="top-to-bottom")[0]		#sorting
correct = 0

for (index, value) in enumerate(np.arange(0, len(questionCnts), 5)):		#left to right {batch 5}
	cnts= contours.sort_contours(questionCnts[value:value+5])[0]

	bubbled= None
	#bubble check
	for (j, c) in enumerate(cnts): 		#mask with bitwise_and
		mask=np.zeros(thresh.shape, dtype="uint8")
		cv2.drawContours(mask,[c], -1,255,-1)

		mask= cv2.bitwise_and(thresh, thresh, mask=mask)
		total= cv2.countNonZero(mask)		#number of non zero pixels

		if bubbled is None or total > bubbled[0]:
			bubbled = (total, j)

	color = (0, 0, 255) 		 #red
	k = answer_key[index]

	if k == bubbled[1]:
		color = (0, 255, 0) 		 #green
		correct += 1
	cv2.drawContours(paper, [cnts[k]],-1, color, 3)


#print (correct)
score = (correct / 5.0) * 100
print("Score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Orignal Image", image)
cv2.imshow("Result", paper)
cv2.waitKey(0)




