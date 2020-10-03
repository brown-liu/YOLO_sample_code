import cv2

cam=cv2.VideoCapture(0)
while True:
    success,img=cam.read()
    cv2.imshow("hello",img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
