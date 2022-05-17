import cv2
import numpy as np

def preprocessImg(imgl):

    img = cv2.imread(imgl) # read img
    img = cv2.resize(img, (1026, 1026)) # resize to (1026,1026)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #turn into gray
    grayImg = cv2.GaussianBlur(grayImg, (11, 11), 0)  #blur to reduce noise

    return img, grayImg


def transformImage(imgl):
    img, grayImg = preprocessImg(imgl)
    thresholdImg = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,5,2)
    kernel = np.array([0,1,0,1,1,1,0,1,0],dtype=np.uint8).reshape(3,3)
    dilationImg = cv2.dilate(thresholdImg, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(dilationImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maxContour = max(contours,key = cv2.contourArea)
    e = 0.1 * cv2.arcLength(maxContour,True)
    finalContour = cv2.approxPolyDP(maxContour, e, True).squeeze().tolist()
    cortour_list = sorted(finalContour, key=lambda c: c[1]) # sort by y values
    finalList = sorted(cortour_list[:2])+sorted(cortour_list[2:]) # sort by x values
    corner_position = np.float32(finalList)
    image_position = np.float32([[0, 0], [1026, 0], [0, 1026], [1026, 1026]])
    transform = cv2.getPerspectiveTransform(corner_position, image_position)

    ori_puzzle = cv2.warpPerspective(img,transform,(1026,1026))
    inputPuzzle = cv2.warpPerspective(dilationImg,transform,(1026,1026))
    dilation_puzzle = cv2.dilate(inputPuzzle, kernel, iterations=1)

    return ori_puzzle, dilation_puzzle

def checkZero(crop_puzzel, crop_ori, x, y):
    x1, y1 = crop_puzzel.shape[1], crop_puzzel.shape[0]
    startx = x1 // 2 - (x // 2)
    starty = y1 // 2 - (y // 2)
    crop_puzzel1 = crop_puzzel[starty:starty + y, startx: startx + x]
    crop_ori1 = crop_ori[starty:starty + y, startx:startx + x]
    contours, _ = cv2.findContours(crop_puzzel1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zero = cv2.imread('/Users/dttai11/sudokuCPU/zero_template.jpg', cv2.IMREAD_GRAYSCALE)
    if len(contours) == 0 or cv2.contourArea(max(contours, key=cv2.contourArea)) < 250:
        return zero
    # else:
    #     return crop_ori
    else:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        d = (h - w) // 2
        c = crop_ori1.shape[0]
        digit = crop_ori1[y:y + h, max(0, x - d):min(c, x + w + d)]  # Save grayscale image crops

        return digit



def puzzel(imgl):
    ori_img, puzzel = transformImage(imgl)
    sudoku = []
    height, width = puzzel.shape[0],puzzel.shape[1]
    cropH, cropW = height//9, width//9
    for y in range(0,height,cropH):
        for x in range(0,width, cropW):
            y1 = y + cropH
            x1 = x + cropW
            crop_puzzel = puzzel[y:y1, x:x1]
            crop_ori = cv2.cvtColor(ori_img[y:y1, x:x1], cv2.COLOR_BGR2GRAY)
            digit = checkZero(crop_puzzel, crop_ori, 82, 82)
            digit = cv2.resize(digit, (32, 32), cv2.INTER_AREA)/255.0
            sudoku.append(digit)
    sudoku_numbers = np.float32(sudoku).reshape(81, 32, 32, 1)
    np.save("/extract_model/preprocess_image/nparray/ans", sudoku_numbers)


puzzel("/Users/dttai11/sudokuCPU/imageTest/test1.jpeg")