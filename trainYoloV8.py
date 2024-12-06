import torch
from ultralytics import YOLO

def DownloadDataV8():
    from roboflow import Roboflow
    rf = Roboflow(api_key="bCFryYsEHyQ3MJEGY9Ml")
    project = rf.workspace("projetos-fensd").project("analise-de-buracos-na-rua")
    version = project.version(1)
    dataset = version.download("yolov8")

def Train():
    # Load a COCO-pretrained YOLO11n model
    model = YOLO("yolov8n.pt")

    print("torch cuda is avalilable: ", torch.cuda.is_available())

    results = model.train(data="C:/virtualD/PyCharmProj/Final/Analise-de-buracos-na-rua-1/data.yaml", epochs=100)  # train the model

if __name__ == "__main__":
    DownloadDataV8()
    Train()