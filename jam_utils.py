def impl_mask(img, roi_mask, print_from_to = False):
    newone = img[roi_mask[2]:roi_mask[3], roi_mask[0]:roi_mask[1]]
    return newone
    #return img[roi_mask[2]:roi_mask[3], roi_mask[0]:roi_mask[1]]

def getMask (width, height, crop_ratio = 0.5):
    rev_ratio = float(2/crop_ratio)
    return [int(width / rev_ratio), int(width * (rev_ratio-1) / rev_ratio), int(height / rev_ratio), int(height * (rev_ratio-1) / rev_ratio)]


def printCVsize(img):
    Global.printCVsizeCnt += 1
    print("CNT :[", Global.printCVsizeCnt, "] width : ", img.shape[1], ", height : ", img.shape[0])

def points_in_ROI(roi, point_number_x, point_number_y, order_xy = False):
    offset_x = roi[0]
    offset_y = roi[2]
    width = (roi[1] - roi[0])
    height = (roi[3] - roi[2])
    interval_x = int(width / (point_number_x-1))
    interval_y = int(height / (point_number_y-1))
    points = np.array([])
    for x in range(point_number_x):
        for y in range(point_number_y):
            x_ = offset_x + interval_x * x
            y_ = offset_y + interval_y*y
            #points.append(Point(offset_x + interval_x * x, offset_y + interval_y*y).to_dict())
            point = None
            if order_xy:
                point = np.array([x_, y_])
            else :
                point = np.array([y_, x_])
            np.vstack((points,point))
    return points