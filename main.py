import numpy as np
import cv2 as cv
import jam_utils as ju

class Global():
    printCVsizeCnt = 0

class Point:
    x = -1.0
    y = -1.0
    border_width = -1.0
    border_height = -1.0
    def __init__(self, _x = -1.0, _y = -1.0, width_border = -1.0, height_border = -1.0):
        self.x = _x
        self.y = _y
        self.border_width = width_border
        self.border_height = height_border
    def isValid(self):
        return bool(not((self.x==-1.0)and(self.y==-1.0)))

    def to_string(self, print_xy_also = "false", parser = ', ', order_xy = True):
        xy_printer = ["x : ", "y : "]
        xy_order = [self.x, self.y]
        if (not order_xy):
            xy_printer = xy_printer.reverse()
            xy_order = xy_order.reverse()
        returner = ""
        if(print_xy_also):
            returner = xy_printer[0]
        returner += str(xy_order[0]) + parser
        if (print_xy_also):
            returner += xy_printer[1]
        returner += str(xy_order[1])
        return returner
    def to_dict(self, order_xy = False):
        return [self.x, self.y] if order_xy else [self.y, self.x]

    def isInBoundary(self, include_border = False):
        if bool(not((self.border_width==-1.0)and(self.border_height==-1.0))):
            print("border not set")
            return False
        return (self.x >=0 and self.y >=0 and ((self.border_width >= self.x) if include_border else (self.border_width >= self.x)) and ((self.border_height >= self.y) if include_border else (self.border_height >= self.y)))





def main():

    p = Point()

    print(p.x, p.y)

    cap = cv.VideoCapture('hand2.mp4')
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    print("old frame size : ",old_frame.size)
    print("old_frame shape : ", old_frame.shape)
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

    #set basic params
    height = int(old_gray.shape[0])
    width  = int(old_gray.shape[1])
    #roi_mask = [int(width/4.0), int(width*3/4.0), int(height/4.0), int(height*3/4.0)]
    roi_mask = ju.getMask(width, height, 1/2)
    print("old_gray shape : ",old_gray.shape)
    print(roi_mask)

    #cv.imwrite("old_frame.jpg", old_frame)
    #cv.imwrite("old_gray.jpg", old_gray)

    old_gray = ju.impl_mask(old_gray, roi_mask)

    print("old gray size : ", old_frame.size)


    #cv.imwrite("old_gray_cropped.jpg", old_gray)

    ju.printCVsize(old_gray)

    #init p0 here
    use_manual_points = True
    if(use_manual_points):
        points_x = 10; points_y = 10
        p0 = ju.points_in_ROI(roi_mask, points_x, points_y)
    else:
        p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    print(p0)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_gray)
    print("mask shape : ", mask.shape)

    initer = True

    frame_num = 0
    while(1):
        print("FRAME : ",frame_num)
        frame_num += 1
        ret,frame = cap.read()
        #print(frame.size)
        #printCVsize(frame)
        frame = ju.impl_mask(frame, roi_mask)

        if(initer): np.zeros_like(frame); initer = False

        #printCVsize(frame)
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)

        '''
        printCVsize(frame)
        print(mask.size)
        print(frame.size)

        print(type(frame), type(mask))
        print("frame shape : ", frame.shape)
        print("mask shape : ", np.zeros_like(frame))
        '''
        frame = frame_gray

        img = cv.add(frame,mask)

        cv.imshow('frame',img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    cv.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()