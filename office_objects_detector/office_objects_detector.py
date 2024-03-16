from ultralytics import YOLO
import cv2 as cv
model = YOLO('yolov8n.pt')
image = cv.imread('office_objects.jpg', 1)
image = cv.resize(image, (0, 0), fx=0.5, fy=0.5)
result = model(image)
count = 0

for i, r in enumerate(result):
    detection = r.boxes.data.tolist()
    classes = r.boxes.cls.tolist()
    names = r.names
    office_object_classes = [24.0, 26.0, 27.0, 28.0, 41.0, 56.0, 57.0, 58.0, 60.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 72.0, 73.0, 74.0, 75.0, 76.0]
    for labels, detected_office_object in zip(classes, detection):
        if labels in office_object_classes:
            count += 1
            label = names[labels]
            x, y, w, h, conf, _ = detected_office_object
            cv.putText(image, label, (int(x), int(y)), cv.FONT_HERSHEY_DUPLEX, 0.5, [50, 50, 255], 1)
            cv.putText(image, str(round(conf, 2)), (int(x), int(h)), cv.FONT_HERSHEY_DUPLEX, 0.5, [50, 50, 255], 1)
            cv.rectangle(image, (int(x), int(y)), (int(w), int(h)), [255, 100, 100], 2)

if count == 0:
    print('There is no office object in the image.')
elif count == 1:
    print('There is 1 office object in the image.')
else:
    print('There are ' + str(count) + ' office objects in the image.')

cv.imshow('Office Objects Detector', image)
cv.waitKey(0)
cv.destroyAllWindows()
