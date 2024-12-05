import cv2
import pandas as pd
import argparse
from ultralytics import YOLO
from tracker import*

# Initialize model and tracker
model = YOLO('E:\Project\Vehicle Counter\Best.pt')
tracker = Tracker()

class_names = ["Motorcycle", "Car", "Bus", "Truck", "background"]

# Argument parsing
parser = argparse.ArgumentParser(description="Vehicle Counter with YOLO")
parser.add_argument("--input", type=str, default="0", help="Path to input video or '0' for webcam")
parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output video")
args = parser.parse_args()

# Open video or webcam
if args.input.isdigit():
    cap = cv2.VideoCapture(int(args.input))  # Webcam input
else:
    cap = cv2.VideoCapture(args.input)  # Video file input

# Configure VideoWriter for saving output in MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(args.output, fourcc, 30.0, (640, 480))

# Dictionary to count vehicles by class
counter_down = {
    "Motorcycle": set(),
    "Car": set(),
    "Bus": set(),
    "Truck": set()
}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (640, 480))
    results = model.predict(frame)
    a = results[0].boxes.data
    a = a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")

    list = []
    objects_info = {}

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_names[d]
        if c in ["Motorcycle", "Car", "Bus", "Truck"]:
            list.append([x1, y1, x2, y2])
            objects_info[(x1, y1, x2, y2)] = c  # Save class info based on bounding box

    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cls = objects_info.get((x3, y3, x4, y4), "unknown")
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.putText(frame, f"{id} Class: {cls}", (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        y = 350
        offset = 7

        if y < (cy + offset) and y > (cy - offset) and cls != "unknown":
            if id not in counter_down[cls]:
                counter_down[cls].add(id)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

    blue_color = (255, 0, 0)
    cv2.line(frame, (0, 300), (640, 300), blue_color, 3)
    
    # Display vehicle counts
    y_position = 40
    for cls, ids in counter_down.items():
        text = f'{cls}: {len(ids)}'
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
        # Tentukan posisi rectangle dan teks
        x_position, y_position = 30, y_position
        rect_start = (x_position - 5, y_position - text_height - 5)  # Margin kecil
        rect_end = (x_position + text_width + 5, y_position + 5)

        # Buat overlay dan tambahkan rectangle putih dengan opacity 20%
        overlay = frame.copy()
        cv2.rectangle(overlay, rect_start, rect_end, (255, 255, 255), -1)  # Rectangle putih solid
        alpha = 0.5  # Opacity 20%
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Tambahkan teks di atas rectangle
        cv2.putText(frame, text, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        y_position += 20

    # Write frame to output video
    out.write(frame)

    cv2.imshow("frames", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
out.release()
cv2.destroyAllWindows()