import cv2
import time

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
        print (one_m_timer_end - one_m_timer_start)
        break

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
file.close()

print ("All count :" , all_count)
print ("Detection count :" , true_count)
