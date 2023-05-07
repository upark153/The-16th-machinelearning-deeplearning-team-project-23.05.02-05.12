# The-16th-machinelearning-deeplearning-team-project-23.05.02-05.12

## 1. 필요한 라이브러리 설치 및 데이터 가져오기
### 1.1 설정
```
!pip install labelme tensorflow opnecv-python matplotlib albumentations
```
> tensorflow-gpu도 설치하길 권장하나, 구글코랩 gpu 대체 가능.

> tensorflow-gpu는 tensorflow 버전에 따라 gpu 통합설치되기도 하고, 버전별 차이점이 존재함.

> opencv-python 오류날 시 , 재설치 하여 오류가 해결 됨.

> labelme는 라벨 작업에 유용한 라이브러리, albumentation은 데이터 증강 시 바운딩 박스 사용.

> Jupyter lab으로 웹캠 데이터를 모은 이후, google colab으로 데이터 학습 시도함.

[Jupyter lab install reference](https://heekangpark.github.io/etc/jupyter-lab)
### Jupyter lab code
### 1.2 OpenCV를 사용하여 이미지 모으기
```
import os
import time
import uuid
import cv2
```
> uuid 는 고유 식별자를 부여한다.
```
uuid.uuid1()
```
> UUID('c946f796-ea1b-11ed-8faa-d46d6d5d1960')
```
IMAGES_PATH = os.path.join('data','images')
number_images = 30
```
> 일부 경로를 정의한다.![image](https://user-images.githubusercontent.com/115389450/236657246-f5ad9e91-1e23-4168-af16-026f6abaa345.png)

```
cap = cv2.VideoCapture(0)
for imgnum in range(number_images):
    print('Collecting image {}'.format(imgnum))
    ret, frame = cap.read()
    imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)
    time.sleep(0.5)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```
> google colab 에서 webcam을 사용하려면 조금 복잡한 코드를 작성해야 한다. 일반 cv2.imshow등을 사용할 수 없다.

> 따라서 jupyter lab을 통해 webcam 이미지를 수집하고, 이후에 google colab gpu 혹은 tensorflow-gpu를
> 사용하는 게 좋다.

### 1.3 lableme로 이미지에 주석 달기
```
!labelme
```
![image](https://user-images.githubusercontent.com/115389450/236658359-927f7728-e14f-4642-8b24-915d29cd876a.png)
1. Open Dir을 눌러서 데이터가 저장된 파일 경로 열기
2. File > Change Ouput Dir 을 눌러서 주석이 담긴 이미지 파일을 저장 할 폴더 지정 ( labels )
3. File > Save Automatically 을 눌러 자동 저장하기
4. Edit > Create Rectangle 을 누른 후 얼굴인식하기 위해 얼굴에 직사각형 그리기
5. 경계상자에 face라는 주석을 적어주고 OK 누르기, 이후 키보드 d를 누르면 다음 이미지로 이동 ( 파일은 .Json으로 저장됨)
6. 반복 작업 ( 흐릿하거나, 얼굴이 짤린 이미지도 사각형 작업을 해주고, 아예 얼굴이 없는 사진은 패스한다. )
7. 혹시라도 경계상자 주석에 face가 아닌 다른 주석이 달린다면, 실제 주석이 달린 파일로 이동하여 지워주면 된다.
8. 다른 주석 삭제하려면, 껐다 키면 되는데, labelme 안에서 가능한지 테스트는 안해봤다.
![image](https://user-images.githubusercontent.com/115389450/236658737-4346a8c3-3373-4033-aa5d-147f691ce9be.png)

------------------------------

### 데이터를 모았고, 사전처리 또한 하였다.
### 이후, 수집된 샘플에서 matplotlib 다음으로 분할한다.
### 교육 테스트 및 검증 파티션, 유효성 검사, 이후 신경망을 구축
### google colab code

## 2. 데이터셋 검토 및 이미지 로딩 기능 구축
### 2.1 Import TF and Deps
```
from google.colab import drive
drive.mount('/content/drive')
```
> 주의할점은, 구글드라이브로 연동하여 사전데이터를 업로드 하고 진행해야한다.
> 그렇지 않으면 재접속할 때마다 업로드한 데이터가 사라질 수 있다.
```
import tensorflow as tf
import cv2
import json
import os
import numpy as np
from matplotlib import pyplot as plt
```
### 2.2 GPU 메모리 증가 제한
```
# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```
```
tf.test.is_gpu_available()
```
> WARNING:tensorflow:From <ipython-input-6-17bb7203622b>:1: is_gpu_available (from tensorflow.python.framework.test_util) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.config.list_physical_devices('GPU')` instead.
True
```
tf.config.list_physical_devices('GPU')
```
> [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
### 2.3 TF 데이터 파이프라인에 이미지 로드
```
images = tf.data.Dataset.list_files('/content/drive/MyDrive/data/images/*.jpg', shuffle=False)
```
```
images.as_numpy_iterator().next() # 지정된 파일 경로 확인, 경로가 중요하다.
```
```
def load_image(x):
    byte_img = tf.io.read_file(x) # 파일 경로를 가져온 후
    img = tf.io.decode_jpeg(byte_img) # 바이트 인코딩 된 이미지를 반환
    return img
```
```
images = images.map(load_image)
```
[tensorflow.Dataset.map](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map)
> ![image](https://user-images.githubusercontent.com/115389450/236659669-55d5b755-a2dc-41be-80c1-aaf02b69e8f5.png)
