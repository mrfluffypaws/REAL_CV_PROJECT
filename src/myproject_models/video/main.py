import cv2
import torch
import numpy as np
from torchvision.transforms import functional as F
import math
from IPython.display import Video, display
from torchvision.models.detection import keypointrcnn_resnet50_fpn

# Загрузка модели
model = keypointrcnn_resnet50_fpn(pretrained=True)

# Перевод в режим инференса
model.eval()

# GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Путь к файлам
reference_video = "reference"
my_video = "you_video"

# Ключевые точки и конечности
keypoints = [
    'nose','left_eye','right_eye','left_ear','right_ear',
    'left_shoulder','right_shoulder','left_elbow','right_elbow',
    'left_wrist','right_wrist','left_hip','right_hip',
    'left_knee','right_knee','left_ankle','right_ankle'
]

limbs = [
    [2,0],[2,4],[1,0],[1,3],
    [6,8],[8,10],[5,7],[7,9],
    [12,14],[14,16],[11,13],[13,15],
    [6,5],[12,11],[6,12],[5,11]
]

# Функции для предворительной обработки видео 
# Извлечение координат ключевых точек
def get_pose(frame, model, device, conf_threshold=0.9):
    img_tensor = F.to_tensor(frame).to(device)
    with torch.no_grad():
        out = model([img_tensor])[0]
    if len(out['keypoints'])>0 and out['scores'][0]>conf_threshold:
        kps = out['keypoints'][0][:,:2].cpu().numpy()
        conf = out['keypoints_scores'][0].cpu().numpy()
        return kps, conf
    else:
        return None, None

# Вычисление косинусного сходства
def cosine_distance(p1,p2):
    p1,p2=p1.flatten(),p2.flatten()
    return p1.dot(p2)/(np.linalg.norm(p1)*np.linalg.norm(p2))

# Вычисление взвешенного расстояния
def weighted_distance(p1,p2,conf):
    p1,p2=p1.flatten(),p2.flatten()
    sum1 = 1/np.sum(conf)
    sum2 = 0
    for i in range(len(p1)):
        idx = math.floor(i/2)
        sum2 += conf[idx]*abs(p1[i]-p2[i])
    return sum1*sum2

# Отрисовка скелета
def draw_pose(frame, pose, color):
    for i,j in limbs:
        if i<len(pose) and j<len(pose):
            pt1, pt2 = tuple(map(int,pose[i])), tuple(map(int,pose[j]))
            cv2.line(frame,pt1,pt2,color,2)
            cv2.circle(frame,pt1,4,color,-1)
            cv2.circle(frame,pt2,4,color,-1)

# Изменение размера с сохранением пропорций
def resize_with_aspect(frame, target_width, target_height):
    h, w = frame.shape[:2]
    scale = min(target_width/w, target_height/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((target_height, target_width,3), dtype=np.uint8)
    x_offset = (target_width - new_w)//2
    y_offset = (target_height - new_h)//2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas


# Обработка видео
width, height = 640,480
ref_cap = cv2.VideoCapture(reference_video)
usr_cap = cv2.VideoCapture(my_video)
fps = int(min(ref_cap.get(cv2.CAP_PROP_FPS), usr_cap.get(cv2.CAP_PROP_FPS)))
out = cv2.VideoWriter('comparison.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width*2,height))

frame_id = 0
while True:
    ret1, ref_frame_orig = ref_cap.read()
    ret2, usr_frame_orig = usr_cap.read()
    if not (ret1 and ret2):
        break

    # Сохранение пропорций
    ref_frame = resize_with_aspect(ref_frame_orig,width,height)
    usr_frame = resize_with_aspect(usr_frame_orig,width,height)

    # Получение ключевых точек из измененных кадров
    ref_pose, ref_conf = get_pose(ref_frame, model, device)
    usr_pose, usr_conf = get_pose(usr_frame, model, device)

    if ref_pose is not None and usr_pose is not None:
        # Вычисление метрик
        cos_sim = cosine_distance(ref_pose, usr_pose)
        w_dist = weighted_distance(ref_pose, usr_pose, ref_conf)

        # Отрисовка скелетов
        draw_pose(ref_frame, ref_pose,(0,0,255))# Эталон
        draw_pose(usr_frame, usr_pose,(0,255,0))# Тест

        # Вывод метрик 
        text = f"Cosine: {cos_sim:.3f} | Weighted: {w_dist:.3f}"
        cv2.putText(ref_frame,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),4)
        cv2.putText(ref_frame,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    # Объединение видео 
    combined = np.hstack((ref_frame,usr_frame))
    out.write(combined)

    frame_id += 1
    if frame_id % 10 == 0:
        print(f"Processed frame {frame_id}")

ref_cap.release()
usr_cap.release()
out.release()
print("Видео сохранено: comparison.mp4")

# Сохранение видео
from google.colab import files
files.download("comparison.mp4")
