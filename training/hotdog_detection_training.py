from ultralytics import YOLO

ROOT_DIR = "/Users/joshua_zhang/Coding Projects/UofT/aUToronto/training"  # Change this to your project root directory

model = YOLO('yolo11n.pt')

results = model.train(
    data=f"{ROOT_DIR}/data.yaml",  # dataset config
    epochs=1,                    # train longer
    imgsz=640,
    plots=True
)

model.export(format='pt', project=f'{ROOT_DIR}/prediction', name='hotdog_model')  # export the model to PyTorch format