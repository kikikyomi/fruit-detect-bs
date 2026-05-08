from ultralytics import YOLO

m = YOLO(r"D:\毕设\模型\yolo\best (1).pt")
print(m.names)
