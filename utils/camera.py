import cv2
from ultralytics import YOLO
model = YOLO("best.pt")
    

def gen_img(url):
    camera = cv2.VideoCapture(url)
    while True:
        success, frame = camera.read()
        res = model(frame)
        for r in res:
        #print(r.boxes)
            for (i, box) in enumerate(r.boxes.xyxy):    
                if (r.boxes.conf[i] < .69): continue
                #print(box)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")