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
images = tf.data.Dataset.list_files('/content/drive/MyDrive/data/images/*.jpg', shuffle=False) # 여기서 shuffle을 해제하면 랜덤 이미지를 아래에서 결과값으로 얻는다.
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

```
images.as_numpy_iterator().next()
```
> ![image](https://user-images.githubusercontent.com/115389450/236662061-45a36535-ba72-4dbb-a661-dc3e828e7a4d.png)
```
type(images)
```
> tensorflow.python.data.ops.map_op._MapDataset

### 2.4 matplotlib로 원시 이미지 보기
```
image_generator = images.batch(4).as_numpy_iterator() # 4개의 집합으로 4개를 시각화 하기
```
[tensorflow.Dataset.batch](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch)
>![image](https://user-images.githubusercontent.com/115389450/236662274-201c3c9d-94f4-4ea9-82a6-2229299c835e.png)

```
plot_images = image_generator.next() # 매번 새로운 데이터 배치로 돌아온다.
```
```
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, image in enumerate(plot_images):
    ax[idx].imshow(image)
plt.show()
```
> ![image](https://user-images.githubusercontent.com/115389450/236662366-a46ee3d8-5966-4f11-b0f2-e3da3f8891bd.png)

### tensorflow 데이터 파이프라인을 사용하므로, 메모리를 제한한다.
## 3. 증강되지 않은 데이터 분할
### 3.1 수동으로 훈련 , 테스트 , 검증 데이터 분할하기 위해 폴더를 만든다.
> ![image](https://user-images.githubusercontent.com/115389450/236664249-9f0d8978-0300-4d18-bbc2-79509f2aa007.png)
> ![image](https://user-images.githubusercontent.com/115389450/236664355-07158f8a-0264-4a55-bbc6-c41f259b8cbf.png)
#### 90*.7 = 62.9999
#### 90개의 수집데이터에서, 70% 즉 63개의 이미지를 훈련 데이터로 사용한다.
#### 90*.15 = 13.5 
#### 90개의 수집데이터에서, 15%,15% 즉 14개와 13개의 이미지를 테스트, 검증 데이터로 사용한다.
> 그다지 과학적이지 않는 방법이지만, 현재 실습 단계에서 시도하는 방법이다.
> 90개의 데이터 중 70%인 63개 이미지를 무작위로 선별한다. -> train images

> 90개의 데이터 중 15%,15%인 14개 이미지, 13개 이미지를 선별 -> test, val images

### 3.2 일치하는 레이블 이동
```
for folder in ['train','test','val']:
    for file in os.listdir(os.path.join('drive','MyDrive','data', folder, 'images')):

        filename = file.split('.')[0]+'.json'
        existing_filepath = os.path.join('drive','MyDrive','data', 'labels', filename)
        if os.path.exists(existing_filepath):
            new_filepath = os.path.join('drive','MyDrive','data',folder,'labels',filename)
            os.replace(existing_filepath, new_filepath)
```
## 4.Albumentation을 사용하여 이미지 및 레이블에 이미지 확대 적용
### 4.1 변환 파이프라인 Albumentation설정
```
import albumentations as alb
```
[albumentation reference](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/)
> ![image](https://user-images.githubusercontent.com/115389450/236669984-67fce28a-c8af-4496-93b2-a96dc4a65609.png)

```
img = cv2.imread(os.path.join('drive','MyDrive','data','train','images','000497ff-ea22-11ed-ba64-d46d6d5d1960.jpg'))
```
```
img.shape
```
> (480, 640, 3) # 기존 이미지 픽셀이 480x640이므로 아래에서 자를 때 450x450이 가능한 것
```
augmentor = alb.Compose([alb.RandomCrop(width=450, height=450), # 여기 최소값에 주의
                         alb.HorizontalFlip(p=0.5), 
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2),
                         alb.RGBShift(p=0.2),
                         alb.VerticalFlip(p=0.5)],
                        bbox_params=alb.BboxParams(format='albumentations', # 여기형식주의
                                                   label_fields=['class_labels']))
```
> ![image](https://user-images.githubusercontent.com/115389450/236670355-ccaabeef-d83d-4d55-b25b-a423550f3d36.png)
> bbox_params=alb.BboxParams(format='albumentations', 위 이미지에서 형식마다 사이즈가 다르다는 것을 보여줌
