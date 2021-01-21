# Before Start it...
이전에 작성한 face_eye_detection.py 코드와 다른 점이 있다면 일단 지금 여기 있는 코드가 최종본이다. 전에는 오직 얼굴 인식만을 위해서 제작한 코드라면 이 코드는 내가 중학교 2학년때 과학 탐구 보고서를 위해서 작성한 코드이다. 

해당 보고서는 개인적인 이유로 업로드는 어렵다. 아무튼 그 점을 알고 코드를 봐주길.
## face_eye_detection.py (Final Code)
일단 전체 코드는 다음과 같다.
```python
import sys
import numpy as np
import cv2
import time
import csv
import logging
import datetime

all_count = 0
true_count = 0

def mylog(start_time, end_time, count):
    print ('st:%s, et:%s, cnt:%s', start_time, end_time, count)

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)


file = open('./result/res_25.csv', 'w')

one_m_timer_start = time.time()
while 1:
    s = time.clock()
    #sTimeTmp = datetime.datetime.now()
    #s = sTimeTmp.strftime('%H : %M : %S ')
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    all_count = all_count + 1

    for (x,y,w,h) in faces:
        #s = time.time()
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        true_count = true_count + 1
        e = time.clock()
        msg = str(s) + ',' + str(e) + ',' + str(e-s) + ',' + str(true_count) +'\n'
        file.write(msg)

        mylog('','',msg)



    one_m_timer_end = time.time()
    #print str(one_m_timer_end - one_m_timer_start)
    if( one_m_timer_end - one_m_timer_start > 10 ):
        print one_m_timer_end - one_m_timer_start
        break

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
file.close()

print "All count :" , all_count
print "Detection count :" , true_count
```
이 코드도 하나하나 따로따로 설명을 하겠다. 이전 문서와 겹치는 부분은 똑같으니 복붙으로 대체.. ~~너무 많아!!~~
```python
import sys
import numpy as np
import cv2
import time
import csv
import logging
import datetime
```
패키지가 많이 쓰였다. 기본적인 것들도 있지만 로깅을 하기 위해서 쓰인 것들이 대다수.
```python
all_count = 0
true_count = 0

def mylog(start_time, end_time, count):
    print ('st:%s, et:%s, cnt:%s', start_time, end_time, count)

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)


file = open('./result/res_25.csv', 'w')

one_m_timer_start = time.time()
```
__all_count__ 와 __true_count__ 같은 경우에는 얼굴의 갯수를 세기 위함이다. __all_count__ 는 시도한 횟수, __true_count__ 는 얼굴이 인식 된 횟수이다. __mylog__ 함수는 현재 작동되는 로그를 프린트 해주는 함수이다. <br>

file는 결과물들을 저장하기 위한 csv 파일을 읽어온다. 작성을 위해서 갖고 온다. <br>

face_cascade는 얼굴인식을 위한 XML(Haar Cascade Classifier) 를 갖고 오는 분류기 이다. __one_m_timer_start__ 는 작동되는 시간을 체크 하기 위함이다. time.time()를 코드 처음에 작성해주면 시간 측정이 가능하다. <br>

```python
while 1:
    s = time.clock()
    #sTimeTmp = datetime.datetime.now()
    #s = sTimeTmp.strftime('%H : %M : %S ')
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    all_count = all_count + 1
```
해당 코드는 계속 작동이 되야 함으로 __while 1:__ 을 활용하였다. s는 현재 시간을 갖고 온다. ret, img는 기본 웹캠에서 읽어오는 카메라 값을 갖고 오기 위함이다. <br>

gray는 img에서 읽어오는 화면을 Grayscale로 변환을 한 모습이다. __cv2.COLOR_BGR2GRAY__ 를 해주는 이유는 빠른 속도 처리를 위함이다. 사람이 보이기에는 사실 흑백보다는 색이 들어간게 편할수는 있다. 하지만 이건 사람을 위한 코드가 아니다. 컴퓨터가 보는 관점에서는 컬러가 들어가면 숫자가 커진다. 수학에서도 숫자 많고 문자 많으면 귀찮고 계산 오래 걸리듯 컴퓨터도 빠른 연산을 위해서는 숫자가 작은게 좋다. 아 참고로 해당 함수는 다양하게 있다. __COLOR_BGR2RGB__ 도 있으며 __HSV__ 등등 여러개가 있다. 매우 중요하기 때문에 반드시 이 부분은 공부해보시길 바란다.<br>

faces는 맨 위에서 읽어온 Haar Cascade를 통해서 얼굴 인식을 해준다. 인식을 해줄때 컴퓨터가 보(?) 얼굴은 Grayscale로 변환을 해준 화면이다. openCV를 지속적으로 하게 되면 __detectMultiScale__ 라는 함수가 굉장히 익숙할 것이다. 개인적으로 중요한 함수들은 공식 문서를 번역기를 쓰던 뭘 하든 읽어보는 걸 추천한다. 해당 함수에 들어간 인자 값들은 순서대로 __image__, __scaleFactor__, __minNeighbors__ 이렇게 총 3개가 있다.<br> 
```python
    for (x,y,w,h) in faces:
        #s = time.time()
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        true_count = true_count + 1
        e = time.clock()
        msg = str(s) + ',' + str(e) + ',' + str(e-s) + ',' + str(true_count) +'\n'
        file.write(msg)

        mylog('','',msg)
```
이 코드가 얼굴인식에 있어서 가장 핵심 부분이다. <br>

for 문에서 x, y, w, h를 쓰는 이유는 얼굴인식을 위해서는 4개의 점이 필요하기 때문이다. 이유는 직사각형을 하나 그리기 위해서는 총 4개의 좌표를 찍어 줘야 한다. 그러면 (x, y) 를 직사각형의 시작점(좌측 상단) 이라고 하면 끝점(우측 하단)의 좌표는 (x + w, y + h)로 된다. 여기서 말하는 faces는 위에서 말한 얼굴인식 분류기에서 갖고 오는 얼굴인식 결과값을 말한다.<br> 

cv2.rectangle 함수를 통해 컬러로 된 화면에 얼굴이 인식이 된 부분에 그려준다. ROI 라는 것은 Rectanglar region Of Interest (관심 영역 부분) 을 말한다. 특정 부위를 직사각형으로 영역을 표시한다 뭐 이런거다. roi_gray에 관심 영역 (여기서는 얼굴로 해당 되는 부분) 을 gray로 부터 갖고 온다. roi_color도 마찬가지로 원본 사진에서 찾은 부분을 영역 표시를 해준다.참고로 OpenCV에서 읽어오는 image들은 모두 Mat type이다. 

true_count는 얼굴이 인식이 된 경우에 수를 추가해준다. e는 인식이 되었을때의 시각, msg는 얼굴이 인식이 된 값을 한번에 저장하는 String 값이다. 그리고 위에서 갖고온 csv 파일에 msg 값을 작성해준다. mylog를 통해 msg 값을 출력한다.
```python
    one_m_timer_end = time.time()
    #print str(one_m_timer_end - one_m_timer_start)
    if( one_m_timer_end - one_m_timer_start > 10 ):
        print one_m_timer_end - one_m_timer_start
        break

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
file.close()

print "All count :" , all_count
print "Detection count :" , true_count
```
one_m_timer_end는 타이머를 종료한 시간을 말한다. 또한 프로그램을 10초가 되며 자동으로 꺼지게 하기 위해서 if 문을 통해서 10초 이상이 되면 마지막 시간을 프린트 하며 프로그램이 꺼지도록 하였다. 

cv2.imshow는 화면 출력을 해준다. 10초 이상 이후에 프로그램이 종료 됨에도 불구하고 cv2.waitKey를 넣어준 이유는 프로그램 실행 중에 혹여나 생긴 오류가 생긴다면 중간에 프로그램 종료를 하기 위해서이다. 참고로 k == 27 이라는 것은 ASCII 값으로 *ESC* 를 뜻한다고 한다.

웹캠 사용을 중단하고 cv2에서 쓰고 있던 Python 화면들도 모두 종료한다. 그리고 마지막으로 안정적으로 파일 작성을 위해서 파일 또한 닫아준다. 최종적으로 얼굴을 세기 위해서 시도 했던 수, 얼굴이 인식이 된 수를 출력해준다.