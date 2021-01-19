## openCV_Face_detection
openCV 의 라이브러리와 Python2.7 버전으로 얼굴인식을 해본다. </br>

## Move New reponsitory.
이 문서는 딱히 쓸 생각이 없었으나, 포토폴리오와 과학탐구 등 여러가지로 다시 오픈(?)을 하게 되었다. </br>
처음으로 깃허브 가입하고 올린 저장소이기도 하고 많이 관리를 안한지라 커밋 내용이 거의 없다 ~~ㅋㅋㅋㅋ~~ </br>
</br>
이후 이 저장소의 커밋은 계속 될 예정이다. 

# How it works?
많이 늦긴 했지만 그래도 차근 차근 설명을 하려고 한다. 사실 이 코드는 중학교때 과학 탐구 보고서를 위한 코드였다. 컴퓨터의 얼굴인식 능력이 과연 빛의 영향을 얼마나 받는가에 대한 탐구 보고서를 위한 코드인데 해당 보고서는 기회가 되면 올리겠다.. <br> <br>

---

## face_eye_detection.py
Haar Cascade를 통해 얼굴을 인식을 하는 코드이다. 사실 이 코드가 최종코드는 아니고 Default코드 같은 느낌이다. 기본적인 요소들은 들어가 있는 그런거.. 최종 코드는 openCV_EYE 안에 있다.
```python
import numpy as np
import cv2

all_count = 0
t_count = 0
```
모듈과 얼굴수를 카운팅 하기 위한 변수를 추가 해준다. 
```python
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
```
Haar Cascade에 필요한 xml 데이터를 끌어와주고 카메라를 열어준다. ~~사실 이 표현이 맞는지는 모르겠다. 뭐 아날로그 식으로 표현하면 연다는 표현도 틀린건 아닌거 같다.~~

```python
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    all_count = all_count + 1

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
	t_count = t_count + 1

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
```
여기는 부분 부분 나눠 가면서 설명 하도록 하겠다.

```python
while 1:
    ret, img = cap.read()
```
ret,img는 반복문에서 지속적으로 cv2.VideoCapture(0) 함수를 통해 기본 웹캠의 화면들을 읽어 온다. 다음 줄에서 Gray로 변환을 해준다.<br> <br>

```python
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    all_count = all_count + 1
```
__cv2.COLOR_BGR2GRAY__ 를 해주는 이유는 빠른 속도 처리를 위함이다. 사람이 보이기에는 사실 흑백보다는 색이 들어간게 편할수는 있다. 하지만 이건 사람을 위한 코드가 아니다. 컴퓨터가 보는 관점에서는 컬러가 들어가면 숫자가 커진다. 수학에서도 숫자 많고 문자 많으면 귀찮고 계산 오래 걸리듯 컴퓨터도 빠른 연산을 위해서는 숫자가 작은게 좋다. 아 참고로 해당 함수는 다양하게 있다. __COLOR_BGR2RGB__ 도 있으며 __HSV__ 등등 여러개가 있다. 매우 중요하기 때문에 반드시 이 부분은 공부해보시길 바란다.<br> <br>

faces는 맨 위에서 읽어온 Haar Cascade를 통해서 얼굴 인식을 해준다. 인식을 해줄때 컴퓨터가 보(?) 얼굴은 Grayscale로 변환을 해준 화면이다. openCV를 지속적으로 하게 되면 __detectMultiScale__ 라는 함수가 굉장히 익숙할 것이다. 개인적으로 중요한 함수들은 공식 문서를 번역기를 쓰던 뭘 하든 읽어보는 걸 추천한다. 해당 함수에 들어간 인자 값들은 순서대로 __image__, __scaleFactor__, __minNeighbors__ 이렇게 총 3개가 있다.<br> <br>
```python
for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
```
x, y, w, h 는 얼굴을 찾기 위한 총 4개의 점이다. 무슨 말이냐면 박스를 그리기 위해서는 총 4개의 점이 필요하다. 즉, 4개의 좌표가 필요하다. 좌측 상하단, 우측 상하단이 필요하기 때문이다. 그 후 __cv2.rectangle()__ 함수로 해당 부분에 직사각형을 그려준다. ROI 라는 것은 Rectanglar region Of Interest (관심 영역 부분) 을 말한다. 특정 부위를 직사각형으로 영역을 표시한다 뭐 이런거다. roi_gray에 관심 영역 (여기서는 얼굴로 해당 되는 부분) 을 gray로 부터 갖고 온다. roi_color도 마찬가지로 원본 사진에서 찾은 부분을 영역 표시를 해준다.참고로 OpenCV에서 읽어오는 image들은 모두 Mat type이다. ~~나도 몰라! 패스~~ <br> <br>

```python
t_count = t_count + 1

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
```
더 이상 뭐 더 없다. 회수를 추가 해주고 해당 이미지를 실시간으로 보여주는 거 이외에는 취소버튼? 그게 끝이다.
<br> <br>