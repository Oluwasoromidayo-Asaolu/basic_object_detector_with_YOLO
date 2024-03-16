from ultralytics import YOLO
import cv2 as cv
model = YOLO('yolov8n.pt')
image = cv.imread('./cars.jpg', 1)
image = cv.resize(image, (0, 0), fx=0.25, fy=0.25)
result = model(image)
count = 0

for i, r in enumerate(result):
    detection = r.boxes.data.tolist()
    classes = r.boxes.cls.tolist()
    names = r.names

    for labels, detected_car in zip(classes, detection):
        if labels == 2.0:
            count += 1
            x, y, w, h, conf, _ = detected_car
            label = names[labels]
            cv.putText(image, label, (int(x), int(y)), cv.FONT_HERSHEY_DUPLEX, 0.7, [50, 50, 255], 1)
            cv.putText(image, str(round(conf, 2)), (int(x), int(h)), cv.FONT_HERSHEY_DUPLEX, 0.7, [50, 50, 255], 1)
            cv.rectangle(image, (int(x), int(y)), (int(w), int(h)), [255, 50, 50], 2)

if count == 0:
    print('There is no car in the image.')
elif count == 1:
    print('There is 1 car in the image.')
else:
    print('There are ' + str(count) + ' cars in the image.')

cv.imshow('Car Detector', image)
cv.waitKey(0)
cv.destroyAllWindows()
