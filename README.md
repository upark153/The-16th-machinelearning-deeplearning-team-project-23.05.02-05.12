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

### Jupyter lab code
### 1.2 OpenCV를 사용하여 이미지 모으기
```
import os
import time
import uuid
import cv2
```
