import pandas as panda
import cv2

initial_state = None

data_frame = panda.DataFrame(columns=["Initial", "Final"])

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    
    var_motion = 0
    
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray_frame = cv2.GaussianBlur(gray_image, (21, 21), 0)
    
    if initial_state is None:
        initial_state = gray_frame
        continue
    
    # tính toán sự khác nhau của initial state và gray_frame
    differ_frame = cv2.absdiff(initial_state, gray_frame)
    
    thresh_frame = cv2.threshold(differ_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
    
    cont,_ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cur in cont:
        if cv2.contourArea(cur) < 10000:
            continue
        
        var_motion = 1
        cur_x, cur_y, cur_w, cur_h  = cv2.boundingRect(cur)
        
        cv2.rectangle(frame, (cur_x, cur_y), (cur_x + cur_w, cur_y + cur_h), (0, 255, 0), 3)

    
    cv2.imshow("Gray Frame", gray_frame)
    cv2.imshow("Difference Frame", differ_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()