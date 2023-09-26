import cv2
import pickle
import numpy as np
import time

estacionamientos = []
with open('espacios.pkl', 'rb') as file:
    estacionamientos = pickle.load(file)

video = cv2.VideoCapture('video.mp4')

start_time = None
end_time = None
tiempo_acumulado = 0
#total_detecciones = 0 

while True:
    check, img = video.read()
    imgBN = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgTH = cv2.adaptiveThreshold(imgBN, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgTH, 5)
    kernel = np.ones((5, 5), np.int8)
    imgDil = cv2.dilate(imgMedian, kernel)

    for i, (x, y, w, h) in enumerate(estacionamientos):
        espacio = imgDil[y:y+h, x:x+w]
        count = cv2.countNonZero(espacio)
        cv2.putText(img, str(count), (x, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

        if count < 7300:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if start_time is None:
                start_time = time.time()
        elif count >= 7300 and start_time is not None:
            end_time = time.time()
            duration = end_time - start_time
            tiempo_acumulado += duration
            start_time = None
            #total_detecciones += 1  
        cv2.putText(img, f"Detector {i + 1}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    tiempo_acumulado = tiempo_acumulado / 60.0  

    cv2.putText(img, f"Tiempo de deteccion: {tiempo_acumulado:.3f} min", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #cv2.putText(img, f"Total de detecciones: {total_detecciones}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('video', img)
    #cv2.imshow('video TH', imgTH)
    #cv2.imshow('video Median', imgMedian)
    #cv2.imshow('video Dilatada', imgDil)
    cv2.waitKey(10)

    if cv2.waitKey(100) & 0xFF == ord(' '):
        break

video.release()
cv2.destroyAllWindows()
