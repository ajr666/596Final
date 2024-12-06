from roboflow import Roboflow
rf = Roboflow(api_key="bCFryYsEHyQ3MJEGY9Ml")
project = rf.workspace("projetos-fensd").project("analise-de-buracos-na-rua")
version = project.version(1)
dataset = version.download("yolov11")