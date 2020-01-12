import os
import cv2
from tqdm import tqdm

model = "1578829545/"

if not os.path.exists("videos/"):
    os.mkdir("videos")

fourcc = cv2.VideoWriter_fourcc(*"H264")
out = cv2.VideoWriter(f"videos/{model.replace('/', '')}.mp4", fourcc, 30.0, (120,120))

for i in tqdm(range(0, len(os.listdir(model)))):
    path = model+f"epoch{i}.jpg"
    img = cv2.imread(path)
    out.write(img)

out.release()