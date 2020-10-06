import cv2
import numpy as np

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
model = "tiny.weights"
config = 'tiny.cfg'
yoloNet = cv2.dnn.readNetFromDarknet(config, model)
ln = yoloNet.getLayerNames()
ln = [ln[i[0] - 1] for i in yoloNet.getUnconnectedOutLayers()]

labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

conf = 0.2
threshold = 0.1
(W, H) = (None, None)

while True:
    ret, frame = cap.read()
    if W is None or H is None: (H, W) = frame.shape[:2]
    yoloNet.setInput(cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False))
    networkOutput = yoloNet.forward(ln)

    boxes = []
    confidences = []
    classIDs = []
    for output in networkOutput:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > conf:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, threshold)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.2f}%".format(LABELS[classIDs[i]], confidences[i]*100)
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,1.5, color, 3)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()