import torch
from ultralytics import YOLO

def Train():
    # Load a COCO-pretrained YOLO11n model
    model = YOLO("yolo11n.pt")

    print("torch cuda is avalilable: ", torch.cuda.is_available())

    results = model.train(data="C:/virtualD/PyCharmProj/Final/Analise-de-buracos-na-rua-1/data.yaml", epochs=100)  # train the model

if __name__ == "__main__":
    Train()