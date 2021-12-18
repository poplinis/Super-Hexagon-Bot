import numpy as np
import cv2
from mss import mss
import time
import os

import win32api
import win32gui

# TODO: Find actual center
# TODO: Analyze cases where it fails to locate player
# TODO: Figure out how to draw all the debug stuff once code is functionalized
        #cv2.line(img, (0, playerLimitBottom), (monitor["width"], playerLimitBottom), colors['Blue'])
        #cv2.line(img, (0, playerLimitTop), (monitor["width"], playerLimitTop), colors['Blue'])
        #cv2.line(img, (0, centerHexLimitBottom), (monitor["width"], centerHexLimitBottom), colors['LightBlue'])
        #cv2.line(img, (0, centerHexLimitTop), (monitor["width"], centerHexLimitTop), colors['LightBlue'])
        #cv2.drawContours(img, [contour_playerSimple], 0, (0, 0, 255), 3)
        #for i in range(len(midpoints)):
        #    cv2.circle(img, (midpoints[i, :]), 4, (0, 255, 0), -1)

# Dictionary of color codes to pull from, uses BGR to match OpenCV
colors = {'Blue' : (255, 0, 0),
          'Green' : (0, 255, 0),
          'Red' : (0, 0, 255),
          'LightBlue' : (255, 255, 0),
          'Yellow' : (0, 255, 255),
          'Pink' : (255, 0, 255),
          'Teal' : (128, 128, 0),
          'Purple' : (128, 0, 128),
          'Gold' : (0, 128, 128),
          'Cornflower' : (255, 128, 128),
          'LightGreen' : (128, 255, 128),
          'LightRed' : (128, 128, 255),
          'SkyBlue' : (255, 255, 128),
          'LightPink' : (255, 128, 255)}

# Function to bring the super hexagon game window to the foreground, needed so keyevents
# take place correctly. Should be called close to startup. Doesn't work (yet) if window
# is minimized
def activateSuperHexagon():
    win32gui.SetForegroundWindow(win32gui.FindWindow(None, "Super Hexagon"))

# Functions to control the player's movement. Left is defined as counter-clockwise
# and Right is defined as clockwise
# Reference: https://gist.github.com/chriskiehl/2906125
def moveLeft(): # aka counter-clockwise
    win32api.keybd_event(ord("A"), 0, 0, 0)
    time.sleep(0.1)
    win32api.keybd_event(ord("A"), 0, 2, 0) # 2 = win32con.KEYEVENTF_KEYUP

def moveRight(): # aka clockwise
    win32api.keybd_event(ord("D"), 0, 0, 0)
    time.sleep(0.1)
    win32api.keybd_event(ord("D"), 0, 2, 0) # 2 = win32con.KEYEVENTF_KEYUP

def init():
    pass

# Capture the game area, return an OpenCV mat (really a numpy array)
def getScreenCap(monitor, mssObject):
    return np.array(mssObject.grab(monitor))

def getThresholdImage(image, thresh_value):
    # Apply thresholding, if base_color is close to white then do an inverted threshold to keep
    # Deadspace black and objects of interest white
    if thresh_value < 200:
        ret, thresholded_image = cv2.threshold(image, thresh_value+25, 255, cv2.THRESH_BINARY)
    else:
        ret, thresholded_image = cv2.threshold(image, thresh_value-45, 255, cv2.THRESH_BINARY_INV)
    
    return thresholded_image

def getContours(image):
    # Find contours in image, used to detect the player and obstacles
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Returns the xy coords of the player if found, otherwise returns false
def findPlayer(contours):
    # Vertical limits on searching for the player
    # TODO: Make these not magic numbers
    playerLimitBottom = 275
    playerLimitTop = 105

    playerIndex = False

    # Height method
    # Attempt to detect the player vertical size
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

    # If the player has been found                     
    if type(playerIndex) != bool:
        # Select the vertex list of contours that represent the player
        contour_player = contours[playerIndex]
        # Epsilon used for approximating the polygon, maximum distance between the original curve and its approximation
        epsPlayer = 0.1*cv2.arcLength(contour_player, True)
        # Use the RDP Algorithm to simplify the contour
        contour_playerSimple = cv2.approxPolyDP(contour_player, epsPlayer, True)

        # Use image moments to find the center of the player
        M = cv2.moments(contour_player)
        if int(M["m00"]) == 0:
            cX = 0
            cY = 0
        else:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        playerCoords = np.array([cX, cY])
        return playerCoords, contour_player
    else:
        global fails
        fails += 1
        return False, None

# Returns the xy coords of the centerHex if found, otherwise returns false
# Also returns the simple contours for the midpoint function to use
def findCenterHex(contours):
    centerHex = False
    centerHexLimitBottom = 245
    centerHexLimitTop = 135

    # Height method
    # Attempt to detect the player and the center hexagon based on their size
    # For each contour...
    for i in range(len(contours)): 
        current_contour = np.squeeze(np.array(contours[i]), axis=1) # Squeeze vertices down to a nx2 array, 
                                                                    # each row is a point, coloumn 0 is x, column 1 is y

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

    if type(centerHex) != bool:
        # Select the vertex list of contours that represent the center hexagon
        cntCenterHex = contours[centerHex]
        # Epsilon for the RDP Algorithm
        epsHex = 0.03*cv2.arcLength(cntCenterHex, True)
        # Simplify contour of center hexagon
        cntCenterHexSimple = cv2.approxPolyDP(cntCenterHex, epsHex, True)
        cntCenterHexSimple = np.squeeze(cntCenterHexSimple)


        # Use image moments fo find the center of the hexagon
        M = cv2.moments(cntCenterHex)
        if int(M["m00"]) == 0:
            cX = 0
            cY = 0
        else:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        print("Center at ({}, {})".format(cX, cY))
        return np.array([cX, cY]), cntCenterHexSimple
    else:
        return False, None

def getLaneMidpoints(cntCenterHexSimple):
    # Find the midpoints of each side of the simplified center hex
    midpoints = np.zeros_like(cntCenterHexSimple)
    take_range = range(1, len(cntCenterHexSimple)+1)
    midpoints[:, 0] = (cntCenterHexSimple[:, 0] + np.take(cntCenterHexSimple[:, 0], take_range, mode='wrap')) / 2.0
    midpoints[:, 1] = (cntCenterHexSimple[:, 1] + np.take(cntCenterHexSimple[:, 1], take_range, mode='wrap')) / 2.0
    return midpoints

def getObstacleImages(midpoints, center, img, contour_player, contour_centerHex, thresh1):
    cX, cY = center

    # Draw lines radiating out from center passing through midpoints
    # Also create a mask to search each 'lane' for obstacles
    # This method strikes me as less than optimal, Maybe polar or parametric
    # coordinate approach would be better?
    maskLanes = np.array(len(midpoints)*[np.zeros((np.shape(img)[0], np.shape(img)[1]), dtype=np.uint8)])
    for i in range(len(midpoints)):
        mX = midpoints[i, 0]
        mY = midpoints[i, 1]

        # Label the lanes on the debug image
        cv2.putText(img, str(i), (mX, mY), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

        if mX == cX: #straight up or straight down (undefined slope)
            tempX = cX
            tempY = 1000 * -1 * (mY > cY)
        else:
            tempSlope = (mY-cY)/(mX-cX)
            tempIntercept = cY - tempSlope * cX

            tempX = 1000 * (1 - (2 * (mX < cX))) # If mX is less than cX (to the left of it), then go in the negative direction
            tempY = int(tempSlope * tempX + tempIntercept)

        # Draw obstacle detection lines
        startPoint = (cX, cY)
        endPoint = (tempX, tempY)
        cv2.line(img, startPoint, endPoint, (0, 255, 0), 3)
        cv2.line(maskLanes[i], startPoint, endPoint, 255, 3)

        # Mask out player and center hex before detecting obstacles
        obstacleMask = np.zeros((np.shape(img)[0], np.shape(img)[1]), dtype=np.uint8) 
        cv2.drawContours(obstacleMask, [contour_player], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(obstacleMask, [contour_centerHex], -1, 255, thickness=15)

        cv2.imshow("obstacle mask", obstacleMask) # Debug visualization

        # Detect obstacles (intersections between thresh1 and maskLanse)
        thresh1 = cv2.bitwise_and(thresh1, cv2.bitwise_not(obstacleMask))
        cv2.bitwise_and(thresh1, maskLanes[i], maskLanes[i])
    
    return maskLanes

# TODO: Not sure this is working 100% correct
def getObstacleDistances(maskLanes, midpoints):
    # Detect obstacles in each lane
    obstacleDistances = []
    for i in range(len(maskLanes)):
        obstacleDistances.append([])
        obstacleContours, hierarchy = cv2.findContours(maskLanes[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Determine distances to each obstacle
        if len(obstacleContours) > 0:
            for j in range(len(obstacleContours)):
                currDist = np.min(np.linalg.norm(obstacleContours[j] - midpoints[i, :]))
                obstacleDistances[i].append(int(currDist))

    #print("Obstacle Distances: ", obstacleDistances)
    list(map(list.sort, obstacleDistances[:]))
    return obstacleDistances

def getPlayerLane(playerCoords, midpoints):
    # Determine what lane the player is in
    playerLane = None
    minDist = 50000
    for i in range(len(midpoints)):
        currDist = np.linalg.norm(midpoints[i] - playerCoords)
        if currDist < minDist:
            minDist = currDist
            playerLane = i
    #print("Player in lane ", playerLane)
    return playerLane

# Move the player based on info on current location and upcoming obstacles
def movePlayer(playerLane, obstacleDistances):
    # First round each distance to the nearest 25th to account for variance
    print("Obstacle distances before rounding: ", obstacleDistances)
    for i in range(len(obstacleDistances)):
        if obstacleDistances[i] != []:
            obstacleDistances[i] = list(map(roundTo, obstacleDistances[i], len(obstacleDistances[i])*[100]))

    print("Obstacle distances after rounding: ", obstacleDistances)
            

    # Go to an empty lane if there is one, otherwise
    # Go to the lane with the largest minimum obstacle distance, otherwise
    # TODO: Not sure what the next case is yet

    # Get list of empty lanes
    emptyLanes = []
    for i in range(len(obstacleDistances)):
        if obstacleDistances[i] == []:
            emptyLanes.append(i)

    # If there is an empty lane
    if emptyLanes != []:
        #print("Empty Lanes: ", emptyLanes)

        # Get the distances to each empty lane
        laneDistances = list(map(playerLaneDistance, len(emptyLanes)*[playerLane]), emptyLanes)
        print("Distance to each lane: ", laneDistances)
        list(map(list.sort, laneDistances))

        minLaneDistance = min(laneDistances)
        targetLane = emptyLanes[laneDistances.index(min(laneDistances))]
        print("Target Lane: ", targetLane)
        #print("ObstacleDistances[targetLane]: ", obstacleDistances[targetLane])
        print("Player Lane: ", playerLane)

        #print("Lane Distances: ", laneDistances)
        #print("Shortest distance to an empty lane is:")
        #print(minLaneDistance)
        #print("Which is targetLane of")
        #print(targetLane)

        # Move the player to the empty lane
        if minLaneDistance[0] == 0:
            print("Staying put...")
        elif minLaneDistance[0] < minLaneDistance[1]:
            print("Moving counter-clockwise...")
            moveLeft()
        else:
            print("Moving clockwise...")
            moveRight()

# Return the distances between the playerLane and the targetLane, indx 0 is left, indx is right
def playerLaneDistance(playerLane, targetLane):
    return [(playerLane-targetLane)%6, (targetLane-playerLane)%6] 

# Determine which direction to go based on a tuple input
def moveShortDirection(distanceTuple):
    pass

def roundTo(num, tolerance):
    return round(num/tolerance)*tolerance

def showImages():
    pass

def main():
    os.chdir(".\\pics")
    activateSuperHexagon()

    monitor_num = 1
    t0 = time.perf_counter()
    times = np.array([])    

    counter = 1
    global fails
    fails = 0

    playing = True

    with mss() as sct:
        mon = sct.monitors[monitor_num]
        
        # Define capture area
        # TODO: What are these magic numbers? This'll be fun to figure out
        monitor = {
            'top': mon['top']+30+50,
            'left': mon['left']+1+95,
            'width': 580,
            'height': 420,
            'mon': monitor_num}
        
        while playing:
            now = time.perf_counter()

            # Capture a frame of the game for analysis
            screen_cap = getScreenCap(monitor, sct)

            # Estimate the center of the play area, used to normalize color. The colors in Super Hexagon are
            # Constantly changing, so a sample of the background is used to determine a thresholding value
            image_center = (int(screen_cap.shape[0]/2), int(screen_cap.shape[1]/2))
        
            # Convert frame to grayscale, needs to be done before thresholding because thresholding only works on a 1 channel image
            grayscale_image = cv2.cvtColor(screen_cap, cv2.COLOR_BGR2GRAY)

            # Background color of game, used to threshold for edge detection
            base_color = grayscale_image[image_center]

            # Get the thresholded image
            thresh1 = getThresholdImage(grayscale_image, base_color)

            # Convert this image back to color so that the contours can be visualized in color
            img = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)

            # Get the contours of the image
            contours = getContours(thresh1)

            # If contours exist in the image (make sure we don't try to access an invalid element)
            if len(contours) > 0:
                player, playerContours = findPlayer(contours)
                if type(player) == bool:
                    continue # Skip this iteration if the player isn't found

                centerHex, centerHexSimpleContours = findCenterHex(contours)
                if type(centerHex) == bool:
                    continue # Skip this iteration if the centerHex isn't found

                laneMidpoints = getLaneMidpoints(centerHexSimpleContours)

                obstacleImages = getObstacleImages(laneMidpoints, centerHex, img, playerContours, centerHexSimpleContours, thresh1)

                obstacleDistances = getObstacleDistances(obstacleImages, laneMidpoints)

                playerLane = getPlayerLane(player, laneMidpoints)

                movePlayer(playerLane, obstacleDistances)



            # Draw circle at detected center
            cv2.circle(img, centerHex, 4, (0, 255, 0), -1)

            # Draw the combined obstacle detection image
            obstacles = np.zeros_like(obstacleImages[0])
            for i in range(len(obstacleImages)):
                cv2.bitwise_or(obstacles, obstacleImages[i], obstacles)
            if type(playerContours) != type(None):
                cv2.drawContours(obstacles, [playerContours], -1, 255, thickness=cv2.FILLED)
            cv2.imshow("Obstacles", obstacles)


            #print("Length cntCenterHex: {}".format(len(cntCenterHex)))
    
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


if __name__ == "__main__":
    main()
    