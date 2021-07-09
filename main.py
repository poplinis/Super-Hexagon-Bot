import numpy as np
import cv2
from mss import mss
import time
import os


os.chdir(".\\pics")

# TODO: Find actual center
# TODO: Analyze cases where it fails to locate player

colors = [(255, 0, 0),
          (0, 255, 0),
          (0, 0, 255),
          (255, 255, 0),
          (0, 255, 255),
          (255, 0, 255),
          (128, 128, 0),
          (128, 0, 128),
          (0, 128, 128),
          (255, 128, 128),
          (128, 255, 128),
          (128, 128, 255),
          (255, 255, 128),
          (255, 128, 255),
          (128, 255, 255)]

with mss() as sct:
    monitor_num = 1
    mon = sct.monitors[monitor_num]
     
    monitor = {
        'top': mon['top']+30+50,
        'left': mon['left']+1+95,
        'width': 580,
        'height': 420,
        'mon': monitor_num}
    
    time.sleep(0.0)
    t0 = time.perf_counter()
    times = np.array([])    
    counter = 1
    fails = 0
    #while time.perf_counter()-t0 < 10:
    while 1:
        now = time.perf_counter()
        
        #time.sleep(0.0)
        cap = np.array(sct.grab(monitor))
        center = (int(cap.shape[0]/2), int(cap.shape[1]/2))
        
        #cap = cv2.resize(cap, (187, 120), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
        #blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        blurred = gray
        base_color = blurred[center]
        if base_color < 200:
            ret, thresh1 = cv2.threshold(blurred, base_color+25, 255, cv2.THRESH_BINARY)
        else:
            ret, thresh1 = cv2.threshold(blurred, base_color-45, 255, cv2.THRESH_BINARY_INV)
        
        #cv2.circle(thresh1, (290, 194), 65, (255,255,255), -1)
        #croppedThresh = thresh1[99:279, 200:380]
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
        # if len(contours)>6:
        #     cv2.destroyAllWindows()
        #     break
        
        img = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)
        #img_smol = cv2.cvtColor(croppedThresh, cv2.COLOR_GRAY2BGR)
        haveContours = len(contours)>0
        
        # If contours have been identified in the image
        if haveContours:
            player = False
            centerHex = False
            
            
            paddingMax = 175+200
            paddingMin = 5+99
            
            # Height method
            tempIndex = 0
            for i in range(len(contours)):
                if not (contours[i][:]>paddingMax).any():
                    if not (contours[i][:]<paddingMin).any():
                        tempSpan = np.ptp(contours[i], axis=0)[0]
                        spanX = tempSpan[0]
                        spanY = tempSpan[1]
                        if (spanX < 20 and spanX > 5):
                            #print("Span: {}".format(span))
                            playerIndex = i
                            player = i
                        if (spanX > 40 and spanY > 40):
                            centerHex = i
                            
            if type(player) != bool:
                # P_low = cv2.arcLength(contours[least], True)
                # A_low = cv2.contourArea(contours[least])
                # L_low = len(contours[least])
                
                # print("Edges: {}, Perimeter: {}, Area: {}".format(L_low, P_low, A_low))
                cntPlayer = contours[player]
                epsPlayer = 0.1*cv2.arcLength(cntPlayer, True)
                cntPlayerSimple = cv2.approxPolyDP(cntPlayer, epsPlayer, True)
                cv2.drawContours(img, [cntPlayerSimple], 0, (0, 0, 255), 3)
                M = cv2.moments(cntPlayer)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                print("Player at ({}, {})".format(cX, cY))
                print("Length cntPlayer: {}".format(len(cntPlayer)))
            else:
                print("Player Not Found")
                fails += 1
            
            if type(centerHex) != bool:
                cntCenterHex = contours[centerHex]
                epsHex = 0.03*cv2.arcLength(cntCenterHex, True)
                cntCenterHexSimple = cv2.approxPolyDP(cntCenterHex, epsHex, True)
                M = cv2.moments(cntCenterHex)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                print("Center at ({}, {})".format(cX, cY))
                cv2.circle(img, (cX, cY), 4, (0, 255, 0), -1)
                if len(cntCenterHexSimple) < 7:
                    cv2.drawContours(img, [np.int32(2.5*(cntCenterHexSimple-(cX, cY))+(cX, cY))], 0, (0, 255, 0), 3)
                    cv2.drawContours(img, [np.int32(4.0*(cntCenterHexSimple-(cX, cY))+(cX, cY))], 0, (0, 255, 0), 3)
                    cv2.drawContours(img, [np.int32(6.5*(cntCenterHexSimple-(cX, cY))+(cX, cY))], 0, (0, 255, 0), 3)
                print("Length cntCenterHex: {}".format(len(cntCenterHex)))
            else:
                print("Center hex not found")
                
            
            
            # for i in range(len(contours)):
            #     tempColor = (0, 0, 255)
            #     if not (contours[i][:]>paddingMax).any():
            #         if not (contours[i][:]<paddingMin).any():
            #             if (np.ptp(contours[i], axis=0)[0][0] < 20):
            #                 tempColor = (0, 255, 0)
            #                 print (np.ptp(contours[i], axis=0)[0][0])
            #     cv2.drawContours(img, [contours[i] + [200, 99]], 0, tempColor, 3)
                
        
        cv2.imshow("FPS Test", img)
        
        # if least and haveContours:
        #     cv2.imwrite("contoured_{}_e_{}.png".format(counter, L_low), img)
        #     np.savetxt("least_{}.csv".format(counter), contours[least].squeeze(), fmt='%.2f', delimiter=",")
        
        counter += 1
        curr_fps = 1/(time.perf_counter()-now)
        times = np.append(times, curr_fps)
        print("fps: {}".format(curr_fps))
        
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            print("Success Rate: {}".format((counter-fails)/counter))
            break
        
    cv2.destroyAllWindows()
    np.savetxt("fps_benchmark.csv", times, delimiter="\n")
    
    