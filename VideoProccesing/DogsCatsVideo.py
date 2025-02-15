import cv2
#XML DOSYASINI(EĞİTİLMİŞ MODEL) TANIMLAMA
dog_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')
#VİDEOYU TANIMLAMA
vid = cv2.VideoCapture('Cats.mp4')

#NESNE TANIMLAMA
while True: 
    ret, img = vid.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dogs = dog_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30), maxSize=(150, 150))
    for(x, y, w, h) in dogs:
        cv2.rectangle(img, (x,y), (x+y, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+h]
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(img, 'Cat', (x, y-h), font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    #VİDEOYU OYNATMA
    cv2.imshow('img', img)
    #VİDEOYU KAPATMA
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

vid.release()
cv2.destroyAllWindows()