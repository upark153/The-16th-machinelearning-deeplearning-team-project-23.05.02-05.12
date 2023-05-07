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

### 4.2 OpenCV 및 JSON으로 테스트 이미지 및 주석 로드
```
img = cv2.imread(os.path.join('drive','MyDrive','data','train','images','66d2e54a-ea21-11ed-8bb6-d46d6d5d1960.jpg'))
```
```
with open(os.path.join('drive','MyDrive','data', 'train', 'labels', '66d2e54a-ea21-11ed-8bb6-d46d6d5d1960.json'), 'r') as f:
    label = json.load(f)
```
```
type(label)
```
> dict
```
label
```
```
{'version': '5.2.0.post4',
 'flags': {},
 'shapes': [{'label': 'face',
   'points': [[263.9086294416243, 122.23350253807106],
    [445.6345177664975, 375.02538071065993]],
   'group_id': None,
   'description': '',
   'shape_type': 'rectangle',
   'flags': {}}],
 'imagePath': '..\\images\\66d2e54a-ea21-11ed-8bb6-d46d6d5d1960.jpg',
...
 'imageHeight': 480,
 'imageWidth': 640}
```
```
label['shapes'][0]['points']
```
>[[263.9086294416243, 122.23350253807106],
 [445.6345177664975, 375.02538071065993]] # 좌표 얻기

### 4.3 이미지 해상도와 일치하도록 좌표 추출 및 크기 조정
```
coords = [0,0,0,0] # 좌표 벡터화
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]
```
```
coords
```
> [263.9086294416243, 122.23350253807106, 445.6345177664975, 375.02538071065993] # 원시 파스칼 voc의 좌표
```
coords = list(np.divide(coords, [640,480,640,480])) # 이미지의 너비, 높이에 따라 나누기
```
```
coords
```
> [0.412357233502538, 0.2546531302876481, 0.6963039340101523, 0.7813028764805415] # albumentation 변환 좌표

### 4.4 Albumentation 적용, 결과 보기
```
augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
```
```
augmented
```
> {'image': array([[[ 45,  50,  53],
         [ 49,  54,  55],
         [ 50,  58,  57],
         ..
         [ 47,  54,  47],
         [ 44,  51,  44],
         [ 38,  45,  38]],
        [[ 45,  49,  50],
         [ 50,  55,  56],
         [ 53,  61,  60],
         ..
         [ 35,  42,  35],
         [ 36,  43,  36],
         [ 38,  45,  38]],
        [[ 41,  45,  46],
         [ 44,  49,  50],
         [ 55,  64,  61],
         ..
         [ 37,  44,  37],
         [ 37,  44,  37],
         [ 36,  43,  36]],
        ..
        [[132, 134, 144],
         [132, 134, 144],
         [128, 130, 138],
         ..
         [164, 162, 162],
         [163, 161, 161],
         [160, 158, 158]],
        [[131, 133, 143],
         [132, 134, 144],
         [130, 132, 140],
         ..
         [162, 160, 160],
         [160, 158, 158],
         [159, 157, 157]],
        [[128, 133, 142],
         [130, 132, 142],
         [131, 132, 142],
         ..
         [160, 158, 158],
         [158, 155, 157],
         [158, 155, 157]]], dtype=uint8),
    'bboxes': [(0.3297010716300056,
       0.22216582064297796,
       0.7335363790186127,
       0.7839255499153976)],
     'class_labels': ['face']}
 
```
augmented.keys()
```
> dict_keys(['image', 'bboxes', 'class_labels'])
```
augmented['image'].shape
```
> (450, 450, 3)
```
cv2.rectangle(augmented['image'],
              tuple(np.multiply(augmented['bboxes'][0][:2], [450,450]).astype(int)), # 최소값
              tuple(np.multiply(augmented['bboxes'][0][2:], [450,450]).astype(int)), # 최대값
                    (250,0,0), 2) # 색깔 , 두께
plt.imshow(augmented['image']) # 파란색으로 보여도 실제 파란색이 아니다 opencv때문에 파란색으로 표시됨.
```
>![image](https://user-images.githubusercontent.com/115389450/236671620-a9869e0d-59d9-420b-a6e6-937bc5477a96.png)

### 현재까지 하나의 이미지에만 적용을 하였다.
## 5. 증강 파이프라인 구축 및 실행
### 5.1 증강 파이프라인 실행 ( 모든 이미지에 적용하기 )
> 폴더만들기
> ![image](https://user-images.githubusercontent.com/115389450/236672833-9b36ed7d-0d93-4b53-b75c-a37f22073d83.png)
```
# 하나의 이미지에서 사각형을 그렸지만,
# 모든 데이터가 적용 되도록 해야 한다.
for partition in ['train', 'test', 'val']:
    for image in os.listdir(os.path.join('drive','MyDrive','data', partition, 'images')):
        img = cv2.imread(os.path.join('drive','MyDrive','data', partition, 'images', image))

        coords = [0,0,0.00001,0.00001]
        label_path = os.path.join('drive','MyDrive','data', partition, 'labels', f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)
            
            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            coords = list(np.divide(coords, [640,480,640,480]))
        
        try:
            for x in range(60): # 기본 이미지당 60개의 이미지 생성하기 ( 증강 이미지 )
                augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                cv2.imwrite(os.path.join('drive','MyDrive','aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

                annotation = {}
                annotation['image'] = image

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0:
                        annotation['bbox'] = [0,0,0,0]
                        annotation['class'] = 0
                    else:
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else:
                    annotation['bbox'] = [0,0,0,0]
                    annotation['class'] = 0
                
                with open(os.path.join('drive','MyDrive','aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)
        except Exception as e:
            print(e)

```
### 5.2 증강 이미지를 Tensorflow 데이터 세트에 로드
```
# 5.2 Load Augmented Images to Tensorflow Dataset
train_images = tf.data.Dataset.list_files('/content/drive/MyDrive/aug_data/train/images/*.jpg', shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120, 120)))
train_images = train_images.map(lambda x: x/255)
```
### image를 resize(120,120) 하는 이유는 그것을 더 압축하여, 더 효율적인 신경망 전달
### x를 255로 나누어 0과 1 최종 레이어에 대한 시그모이드 활성화
```
test_images = tf.data.Dataset.list_files('/content/drive/MyDrive/aug_data/test/images/*.jpg', shuffle=False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, (120, 120)))
test_images = test_images.map(lambda x: x/255)
```
```
val_images = tf.data.Dataset.list_files('/content/drive/MyDrive/aug_data/val/images/*.jpg', shuffle=False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x, (120, 120)))
val_images = val_images.map(lambda x: x/255)
```
```
train_images.as_numpy_iterator().next()
```
![image](https://user-images.githubusercontent.com/115389450/236673429-b4fbd803-3457-43d1-a120-e26a86ce1412.png)

## 6. 라벨 준비하기
### 6.1 라벨 로딩 함수
```
# 6. Prepare Labels
# 6.1 Build Label Loading Function
def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)
    return [label['class']], label['bbox']
```
### 6.2 tensorflow 데이터 세트에 라벨 로드
```
train_labels = tf.data.Dataset.list_files('/content/drive/MyDrive/aug_data/train/labels/*.json', shuffle = False)
```
```
train_labels.as_numpy_iterator().next() # 경로 확인
```
> b'/content/drive/MyDrive/aug_data/train/labels/000497ff-ea22-11ed-ba64-d46d6d5d1960.0.json'
```
train_labels = tf.data.Dataset.list_files('/content/drive/MyDrive/aug_data/train/labels/*.json', shuffle = False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
```
> 람다 함수를 사용하여 루프를 통해 각 개별 파일 이름은 tf.py_function 래핑 기능
> 위에서 정의한 load_labels 함수를 사용하고 있다.
```
test_labels = tf.data.Dataset.list_files('/content/drive/MyDrive/aug_data/test/labels/*.json', shuffle = False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
```
```
val_labels = tf.data.Dataset.list_files('/content/drive/MyDrive/aug_data/val/labels/*.json', shuffle = False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
```
```
train_labels.as_numpy_iterator().next()
```
> (array([1], dtype=uint8),
 array([0.12134, 0.2059 , 0.5386 , 0.7407 ], dtype=float16))

## 7. 라벨 및 이미지 샘플 결합
### 7.1 파티션 길이 확인
```
len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images), len(val_labels)
```
> (3780, 3780, 840, 840, 780, 780) # 훈련 샘플 3780, 테스트 840, 검증 780 이미지
### 7.2 최종 데이터 세트 생성 ( 이미지 / 라벨 )
```
# 7.2 Create Final Datasets (Images/Labels)
train = tf.data.Dataset.zip((train_images, train_labels)) # zip 으로 결합.
train = train.shuffle(4000) # 데이터 세트의 크기보다 커야함. ex. 3780 < 4000
train = train.batch(8) # 8개의 이미지와 8개의 라벨
train = train.prefetch(4) # 병목 현상 제거
```
```
test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1300)
test = test.batch(8)
test = test.prefetch(4)
```
```
val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1000)
val = val.batch(8)
val = val.prefetch(4)
```
```
train.as_numpy_iterator().next()[0].shape
```
> (8, 120, 120, 3)  # 이미지 8개, 너비 120 픽셀 120, 3채널
```
train.as_numpy_iterator().next()[1]
```
> (array([[1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1]], dtype=uint8),
 array([[0.2754, 0.191 , 0.6973, 0.7607],
        [0.2396, 0.1869, 0.641 , 0.733 ],
        [0.2974, 0.2334, 0.707 , 0.7783],
        [0.4763, 0.2461, 0.89  , 0.809 ],
        [0.3228, 0.2404, 0.7437, 0.8135],
        [0.1714, 0.2808, 0.5854, 0.758 ],
        [0.    , 0.2585, 0.3806, 0.746 ],
        [0.436 , 0.2039, 0.8555, 0.683 ]], dtype=float16))

### 7.3 이미지 및 주석 보기
```
data_samples = train.as_numpy_iterator()
```
```
res = data_samples.next()
```
```
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx in range(4):
    sample_image = res[0][idx]
    sample_coords = res[1][1][idx]

    cv2.rectangle(sample_image,
                  tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),
                  tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)),
                    (255,0,0), 2)
    ax[idx].imshow(sample_image)
```
> ![image](https://user-images.githubusercontent.com/115389450/236674594-59ef4a7e-c7ae-4b16-9500-4229ed294479.png)

## 8. 기능적 API를 사용하여 딥 러닝 구축
### 8.1 레이어 및 기본 네트워크 가져오기 ( 분류모델(vgg16), 회귀모델 )
```
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16
```
>![image](https://user-images.githubusercontent.com/115389450/236675558-7dcfd89e-9619-436d-9f76-7c8bcdb817ee.png)
>![image](https://user-images.githubusercontent.com/115389450/236675592-12c6b1e9-2d4c-462e-8dbe-27ce291c681a.png)

### 8.2 VGG16 다운로드 ( 인스턴스 만들기 )
```
vgg = VGG16(include_top=False)
```
> Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
58889256/58889256 [==============================] - 4s 0us/step
![image](https://user-images.githubusercontent.com/115389450/236676294-7bc74101-bca0-4bd6-9249-1f33a695fdab.png)
> ![image](https://user-images.githubusercontent.com/115389450/236676237-b1a9979b-48b6-470c-a476-68762929c8b0.png)

```
vgg.summary()
```
> Model: "vgg16"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, None, None, 3)]   0         
                                                                 
 block1_conv1 (Conv2D)       (None, None, None, 64)    1792      
                                                                 
 block1_conv2 (Conv2D)       (None, None, None, 64)    36928     
                                                                 
 block1_pool (MaxPooling2D)  (None, None, None, 64)    0         
                                                                 
 block2_conv1 (Conv2D)       (None, None, None, 128)   73856     
                                                                 
 block2_conv2 (Conv2D)       (None, None, None, 128)   147584    
                                                                 
 block2_pool (MaxPooling2D)  (None, None, None, 128)   0         
                                                                 
 block3_conv1 (Conv2D)       (None, None, None, 256)   295168    
                                                                 
 block3_conv2 (Conv2D)       (None, None, None, 256)   590080    
                                                                 
 block3_conv3 (Conv2D)       (None, None, None, 256)   590080    
                                                                 
 block3_pool (MaxPooling2D)  (None, None, None, 256)   0         
                                                                 
 block4_conv1 (Conv2D)       (None, None, None, 512)   1180160   
                                                                 
 block4_conv2 (Conv2D)       (None, None, None, 512)   2359808   
                                                                 
 block4_conv3 (Conv2D)       (None, None, None, 512)   2359808   
                                                                 
 block4_pool (MaxPooling2D)  (None, None, None, 512)   0         
                                                                 
 block5_conv1 (Conv2D)       (None, None, None, 512)   2359808   
                                                                 
 block5_conv2 (Conv2D)       (None, None, None, 512)   2359808   
                                                                 
 block5_conv3 (Conv2D)       (None, None, None, 512)   2359808   
                                                                 
 block5_pool (MaxPooling2D)  (None, None, None, 512)   0         
                                                                 
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
_________________________________________________________________

### 원본이미지에서 (None, None, None, 512) 이 부분이 변경되거나 조정될 수 있다.

### 8.3 네트워크 인스턴스 빌드 ( 신경망 구축 )
```
# 8.3 Build instance of Network
def build_model(): 
    input_layer = Input(shape=(120,120,3)) # 입력 레이어 지정, 신경망을 구축할 때마다 네트워크가 있어야함

    vgg = VGG16(include_top=False)(input_layer)

    # Classification Model
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)

    # Bounding box model
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)

    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
    return facetracker
```
> ![image](https://user-images.githubusercontent.com/115389450/236676756-7491c29b-b421-4ba9-a3ec-2ed23d8a0cf8.png)
```
train.as_numpy_iterator().next()[1]
```
> 현재 이부분에서 0의 값을 취하는 데이터셋트를 포함을 시키지 못했다. 결과가 어떻게 나올까?
> 그것은 추후에 알아보고, 방법론에 대해 알았으니 수정작업을 해야할듯 싶기도 하다.
(array([[1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1]], dtype=uint8),
 array([[0.6035, 0.2068, 1.    , 0.7573],
        [0.484 , 0.285 , 0.8438, 0.664 ],
        [0.1653, 0.256 , 0.5693, 0.818 ],
        [0.566 , 0.291 , 1.    , 0.878 ],
        [0.51  , 0.1914, 0.9316, 0.7837],
        [0.1643, 0.1665, 0.603 , 0.753 ],
        [0.4426, 0.299 , 0.8623, 0.7783],
        [0.2566, 0.267 , 0.658 , 0.813 ]], dtype=float16))

### 8.4 신경망 테스트
```
facetracker = build_model()
```
```
facetracker.summary()
```
> Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_2 (InputLayer)           [(None, 120, 120, 3  0           []                               
                                )]                                                                
                                                                                                  
 vgg16 (Functional)             (None, None, None,   14714688    ['input_2[0][0]']                
                                512)                                                              
                                                                                                  
 global_max_pooling2d (GlobalMa  (None, 512)         0           ['vgg16[0][0]']                  
 xPooling2D)                                                                                      
                                                                                                  
 global_max_pooling2d_1 (Global  (None, 512)         0           ['vgg16[0][0]']                  
 MaxPooling2D)                                                                                    
                                                                                                  
 dense (Dense)                  (None, 2048)         1050624     ['global_max_pooling2d[0][0]']   
                                                                                                  
 dense_2 (Dense)                (None, 2048)         1050624     ['global_max_pooling2d_1[0][0]'] 
                                                                                                  
 dense_1 (Dense)                (None, 1)            2049        ['dense[0][0]']                  
                                                                                                  
 dense_3 (Dense)                (None, 4)            8196        ['dense_2[0][0]']                
                                                                                                  
==================================================================================================
Total params: 16,826,181
Trainable params: 16,826,181
Non-trainable params: 0
__________________________________________________________________________________________________
```
X, y = train.as_numpy_iterator().next() # X는 이미지, y는 라벨
```
```
X.shape
```
> (8, 120, 120, 3)
```
classes, coords = facetracker.predict(X)
```
```
classes, coords
```
> (array([[0.79747915],
        [0.7802991 ],
        [0.7429593 ],
        [0.60588175],
        [0.5818504 ],
        [0.66104424],
        [0.70686543],
        [0.6533249 ]], dtype=float32),
 array([[0.54703015, 0.436778  , 0.5712255 , 0.6755223 ],
        [0.54020405, 0.42169562, 0.5497299 , 0.6604245 ],
        [0.54898626, 0.46477002, 0.49333876, 0.71200246],
        [0.50955355, 0.39625728, 0.430492  , 0.625896  ],
        [0.46096724, 0.3478772 , 0.47030187, 0.60502297],
        [0.5915168 , 0.43535933, 0.49920946, 0.6382403 ],
        [0.4378142 , 0.3690692 , 0.4548935 , 0.60718215],
        [0.3843374 , 0.37540388, 0.53875136, 0.6154837 ]], dtype=float32))

### 신경망 구축은 8단계까지이다.
## 9.define losses and optimizers ( 손실 및 옵티마이저 정의 )
### 9.1 define optimizer and LR ( 학습 속도를 줄이는 방법 )
#### 1) 배치 수 
```
batches_per_epoch = len(train) # 473
lr_decay = (1./0.75 -1)/batches_per_epoch
```
