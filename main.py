import numpy as np
import cv2
from mss import mss
import time
import os


os.chdir(".\\pics")

# TODO: Find actual center
# TODO: Analyze cases where it fails to locate player

# List of color codes to pull from
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
     
    # Define capture area
    monitor = {
        'top': mon['top']+30+50,
        'left': mon['left']+1+95,
        'width': 580,
        'height': 420,
        'mon': monitor_num}
    
    t0 = time.perf_counter()
    times = np.array([])    

    counter = 1
    fails = 0

    while 1:
        now = time.perf_counter()
        
        # Capture a frame of the game for analysis
        screen_cap = np.array(sct.grab(monitor))

        # Estimate the center of the play area, used to normalize color. The colors in Super Hexagon are
        # Constantly changing, so a sample of the background is used to determine a thresholding value
        center = (int(screen_cap.shape[0]/2), int(screen_cap.shape[1]/2))
        
        # Convert frame to grayscale, needs to be done before thresholding because thresholding only works on a 1 channel image
        gray = cv2.cvtColor(screen_cap, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur, potentially helps with edge detection
        #blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        blurred = gray # Not actually doing the blur right now

        # Background color of game, used to threshold for edge detection
        base_color = blurred[center]

        # Apply thresholding, if base_color is close to white then do an inverted threshold to keep
        # Deadspace black and objects of interest white
        if base_color < 200:
            ret, thresh1 = cv2.threshold(blurred, base_color+25, 255, cv2.THRESH_BINARY)
        else:
            ret, thresh1 = cv2.threshold(blurred, base_color-45, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours in image, used to detect the player and obstacles
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Convert this image back to color so that the contours can be visualized in color
        img = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)

        haveContours = len(contours)>0
        
        # If contours have been identified in the image
        if haveContours:
            player = False
            centerHex = False
            
            # Vertical limits on searching for the player
            # TODO: Make these not magic numbers
            playerLimitBottom = 275
            cv2.line(img, (0, playerLimitBottom), (monitor["width"], playerLimitBottom), colors[0])
            playerLimitTop = 105
            cv2.line(img, (0, playerLimitTop), (monitor["width"], playerLimitTop), colors[0])
            
            # Vertical limits on searching for the center hexagon
            # TODO: Make these not magic numbers
            centerHexLimitBottom = 245
            cv2.line(img, (0, centerHexLimitBottom), (monitor["width"], centerHexLimitBottom), colors[3])
            centerHexLimitTop = 135
            cv2.line(img, (0, centerHexLimitTop), (monitor["width"], centerHexLimitTop), colors[3])
            
            # Height method
            # Attempt to detect the player and the center hexagon based on their size
            # For each contour...
            for i in range(len(contours)): 
                current_contour = np.squeeze(np.array(contours[i]), axis=1) # Squeeze vertices down to a nx2 array, 
                                                                            # each row is a point, coloumn 0 is x, column 1 is y

                # If none of the contour's vertices exceed the top limit...
                if not (current_contour[:, 1]>playerLimitBottom).any(): 
                    # If none of the contour's vertices go below the bottom limit...
                    if not (current_contour[:, 1]<playerLimitTop).any(): 
                        # Get the width and height of the bounding rectangle for the current contour
                        tempSpan = np.ptp(contours[i], axis=0)[0]
                        spanX = tempSpan[0] # Width
                        spanY = tempSpan[1] # Height
                        # Assume the player is between 5 and 20 pixels in the x
                        if (spanX < 20 and spanX > 5):
                            playerIndex = i
                            player = i
                
                # If none of the contour's vertices exceed the top limit...
                if not (current_contour[:, 1]>centerHexLimitBottom).any(): 
                    # If none of the contour's vertices go below the bottom limit...
                    if not (current_contour[:, 1]<centerHexLimitTop).any(): 
                        # Get the width and height of the bounding rectangle for the current contour
                        tempSpan = np.ptp(contours[i], axis=0)[0]
                        spanX = tempSpan[0] # Width
                        spanY = tempSpan[1] # Height
                        # Assume the center hex is greater than 40 pixels in x and y
                        if (spanX > 40 and spanY > 40):
                            centerHex = i

            # If the player has been found                     
            if type(player) != bool:
                # Select the vertex list of contours that represent the player
                contour_player = contours[playerIndex]
                # Epsilon used for approximating the polygon, maximum distance between the original curve and its approximation
                epsPlayer = 0.1*cv2.arcLength(contour_player, True)
                # Use the RDP Algorithm to simplify the contour
                contour_playerSimple = cv2.approxPolyDP(contour_player, epsPlayer, True)

                # Draw the contours in the visualiztion window for debugging
                cv2.drawContours(img, [contour_playerSimple], 0, (0, 0, 255), 3)
                
                # Use image moments to find the center of the player
                M = cv2.moments(contour_player)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                print("Player at ({}, {})".format(cX, cY))
                print("Length contour_player: {}".format(len(contour_player)))
            else:
                print("Player Not Found")
                fails += 1
            
            if type(centerHex) != bool:
                # Select the vertex list of contours that represent the center hexagon
                cntCenterHex = contours[centerHex]
                # Epsilon for the RDP Algorithm
                epsHex = 0.03*cv2.arcLength(cntCenterHex, True)
                # Simplify contour of center hexagon
                cntCenterHexSimple = cv2.approxPolyDP(cntCenterHex, epsHex, True)
                
                # Use image moments fo find the center of the hexagon
                M = cv2.moments(cntCenterHex)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                print("Center at ({}, {})".format(cX, cY))

                # Draw circle at detected center
                cv2.circle(img, (cX, cY), 4, (0, 255, 0), -1)

                # Draw boundaries to potentially detect obstacles
                if len(cntCenterHexSimple) < 7:
                    cv2.drawContours(img, [np.int32(2.5*(cntCenterHexSimple-(cX, cY))+(cX, cY))], 0, (0, 255, 0), thickness=5)
                    cv2.drawContours(img, [np.int32(4.3*(cntCenterHexSimple-(cX, cY))+(cX, cY))], 0, (0, 255, 0), thickness=5)
                    cv2.drawContours(img, [np.int32(6.5*(cntCenterHexSimple-(cX, cY))+(cX, cY))], 0, (0, 255, 0), thickness=5)
                print("Length cntCenterHex: {}".format(len(cntCenterHex)))
            else:
                print("Center hex not found")
        
        # Debugging visualization window
        cv2.imshow("FPS Test", img)
        cv2.imshow("Binarized", thresh1)
        
        counter += 1
        curr_fps = 1/(time.perf_counter()-now)
        times = np.append(times, curr_fps)
        print("fps: {}".format(curr_fps))
        
        # Exit on keypress q on imshow window and display success rate of finding player
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            print("Success Rate: {}".format((counter-fails)/counter))
            break
        
    cv2.destroyAllWindows()
    np.savetxt("fps_benchmark.csv", times, delimiter="\n")
    
    