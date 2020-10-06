#!/usr/bin/env python

import cv2

i=1;
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read(0)
    cv2.imshow('fisheye image', frame)

    if cv2.waitKey(33) == ord('a'):
        print('saving photo number ', i)
        cv2.imwrite('images_gathered/img' + str(i)+'.jpg', frame)
        i = i + 1   

    if cv2.waitKey(33) == ord('q'):
        print('exit from saving photos')
        break

