import cv2
import math

# use which alogrithm to track
whichAlogrithm = "kcf"

# get one new tracker
def getNewTracker(which="kcf"):
    (major, minor) = cv2.__version__.split(".")[:2]
    if int(major) == 3 and int(minor) < 3:
        return cv2.Tracker_create[which]()
    else:
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }
        return OPENCV_OBJECT_TRACKERS[which]()

def cal_Distance(a1, b1, c1, a2, b2, c2):
    z2 = math.sqrt(math.fabs(pow(c2, 2) - pow(a2, 2) - pow(b2, 2)))
    z1 = math.sqrt(math.fabs(pow(c1, 2) - pow(a1, 2) - pow(b2, 2)))
    distance = math.sqrt(pow((a1 - a2), 2) + pow((b1 - b2), 2) + math.pow((z1 - z2), 2))
    print(distance, 'm')
    return distance

def getMoveBox(lastFrame, nowFrame):
    gray = cv2.cvtColor(nowFrame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if lastFrame is None:
        return False, [], gray
    frameDelta = cv2.absdiff(lastFrame, gray)
    thresh = cv2.threshold(frameDelta, 10, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

    allBoxes = []
    wh = nowFrame.shape[0]
    ww = nowFrame.shape[1]
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if wh != h and ww!=w:
            allBoxes.append([x,y,x+w,y+h])

    return True, allBoxes, gray

# judge whether the box has intersect
def hasIntersect(allBoxes, targetBox):
    for oneBox in allBoxes:
        zx = math.fabs(oneBox[0]+oneBox[2]-targetBox[0]-targetBox[2])
        zy = math.fabs(oneBox[1]+oneBox[3]-targetBox[1]-targetBox[3])
        x = math.fabs(oneBox[2]-oneBox[0])+math.fabs(targetBox[2]-targetBox[0])
        y = math.fabs(oneBox[3] - oneBox[1]) + math.fabs(targetBox[3] - targetBox[1])
        if zx<=x and zy<=y:
            return True
    return False

# Calculate the average speed
def calAvgSpeed(disAndtime, frameNum=3):
    nowCount = len(disAndtime)
    if nowCount>frameNum:
        nowCount = nowCount - nowCount % frameNum
    allDis = 0
    allTime = 0
    if nowCount<frameNum:
        frameNum = nowCount
    for i in range(1,frameNum+1):
        allDis = allDis + disAndtime[nowCount-i][0]
        allTime = allTime + disAndtime[nowCount-i][1]
    return allDis/allTime

