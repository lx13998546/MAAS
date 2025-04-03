from ultralytics import YOLO
import time

model = YOLO("./weight/detector.pt")
start_time = time.time()
model.predict("/DATA/FOLDER/",
               save=True, 
               save_txt=False,
               save_crop=True,
               imgsz=640, 
               conf=0.6,
               project='/detector/output/',
               name='project_name',
)
# Stop the timer
end_time = time.time()

# Calculate the total time taken
elapsed_time = end_time - start_time
print(f"Detection process took {elapsed_time:.2f} seconds.")
