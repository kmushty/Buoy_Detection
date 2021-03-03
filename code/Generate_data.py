import sys
import glob
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import math
import copy
from scipy import stats
import random

##ix=0
##iy=0
##click_check=0

class CoordinateSave:
    def _init_(self):
        self.pts = []

    def pic_select(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ix,iy = x,y
##            click_check=1
##            print(x,y)
            self.pts = ix,iy
##        elif event == cv2.EVENT_LBUTTONDBLCLK:
        elif event == cv2.EVENT_RBUTTONDOWN:
            ix,iy = 0,0
##            print(x,y)
            self.pts = ix,iy
            

CdSave = CoordinateSave()

def orange_circle_click(img):
    click_check=0
    cv2.namedWindow('Select Orange Bouy')
    cv2.setMouseCallback('Select Orange Bouy', CdSave.pic_select)
    cv2.imshow('Select Orange Bouy', img)
    cv2.waitKey(0)
    orange_select = CdSave.pts
    print(orange_select)
    cv2.destroyAllWindows()
    return orange_select
    
def green_circle_click(img):
    click_check=0
    cv2.namedWindow('Select Green Bouy')
    cv2.setMouseCallback('Select Green Bouy', CdSave.pic_select)
    cv2.imshow('Select Green Bouy', img)
    cv2.waitKey(0)
    green_select = CdSave.pts
    print(green_select)
    cv2.destroyAllWindows()
    return green_select

def yellow_circle_click(img):
    click_check=0
    cv2.namedWindow('Select Yellow Bouy')
    cv2.setMouseCallback('Select Yellow Bouy', CdSave.pic_select)
    cv2.imshow('Select Yellow Bouy', img)
    cv2.waitKey(0)
    yellow_select = CdSave.pts
    print(yellow_select)
    cv2.destroyAllWindows()
    return yellow_select

def confidence_int(data, confidence):
    a = np.asarray(data)
    mean, sigma = np.mean(a), np.std(a)
    conf_int = stats.norm.interval(confidence, loc=mean, scale=sigma)
    ret=[conf_int,mean,sigma]
##    return conf_int
    return ret


def plot_gauss(mean, sigma, color,title):
    x = list(range(0, 255))
    gauss = ((1 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-np.power(x - mean, 2.) / (2 * np.power(sigma, 2.))))
    plt.title(title)
    plt.plot(gauss, color= color)
    plt.show()
    return gauss


#########################
####Main Training Code
#start Frame Reading

path=os.getcwd()
vidpath = os.path.join(path, 'Videos')
success=False

vidcap = cv2.VideoCapture(vidpath + '/detectbuoy_video.avi')
##success,image = vidcap.read()
count = 0
success=vidcap.isOpened()
img_list=[]

while count<200:
    success,image = vidcap.read()
    cv2.imwrite("/Videos/Sample_Frames/frame%d.jpg" % count, image)     # save frame as JPEG file
    if success==True:
        img_list.append(image)
    count += 1

##np.asarray(img_list)

#end Frame Reading
#########
#start Color Training

train_check = 0
    
print("Color training will begin for the system.....")
print("To select color: Left click on buoy color identified at top of image display")
print("If buoy is partially off screen or not present on image, right click anywhere on the image display")
print("Hit the Space Bar after making selection to proceed. Coordinate selected will appear on command window")

train_img_list=[]
step = 5
for i in range(0, len(img_list), step):
    img_check=img_list[i]
    train_img_list.append(img_check)

np.asarray(train_img_list)
np.asarray(img_list)

orange_pts = []
green_pts = []
yellow_pts = []

orange_circle_train = []
green_circle_train = []
yellow_circle_train = []

orange_to_train=[]
green_to_train=[]
yellow_to_train=[]

orange_count=0
green_count=0
yellow_count=0

folder_name = [path+"/Videos/Training_Frames/Green_Frames", path+"/Videos/Training_Frames/Orange_Frames", path+"/Videos/Training_Frames/Yellow_Frames"]



for k in range(0, len(train_img_list)):
    
    orange_train = copy.deepcopy(train_img_list[k])
    green_train = copy.deepcopy(train_img_list[k])
    yellow_train = copy.deepcopy(train_img_list[k])
    
    orange_select = orange_circle_click(orange_train)
    orange_pts.append(orange_select)
    if orange_select[0]!=0 and orange_select[1]!=0:
        orange_count+=1
        orange_img_crop = orange_train[orange_select[1]-35:orange_select[1]+35,orange_select[0]-35:orange_select[0]+35]
        orange_circle_train.append(orange_img_crop)
        cv2.imwrite(path+"/Videos/Training_Frames/Orange_Frames/Orange_Frame%d.jpg" % orange_count, orange_img_crop)

        image = orange_img_crop
        nx,ny,ch = image.shape
        image = np.reshape(image, (nx * ny, ch))
        orange_to_train.append(image[k,:])
    
    green_select = green_circle_click(green_train)
    green_pts.append(green_select)
    if green_select[0]!=0 and green_select[1]!=0:
        green_count+=1
        green_img_crop = green_train[green_select[1]-35:green_select[1]+35,green_select[0]-35:green_select[0]+35]
        green_circle_train.append(green_img_crop)
        cv2.imwrite(path+"/Videos/Training_Frames/Green_Frames/Green_Frame%d.jpg" % green_count, green_img_crop)

        image = green_img_crop
        nx,ny,ch = image.shape
        image = np.reshape(image, (nx * ny, ch))
        green_to_train.append(image[k,:])
    
    yellow_select = yellow_circle_click(yellow_train)
    yellow_pts.append(yellow_select)
    if yellow_select[0]!=0 and yellow_select[1]!=0:
        yellow_count+=1
        yellow_img_crop = yellow_train[yellow_select[1]-35:yellow_select[1]+35,yellow_select[0]-35:yellow_select[0]+35]
        yellow_circle_train.append(yellow_img_crop)
        cv2.imwrite(path+"/Videos/Training_Frames/Yellow_Frames/Yellow_Frame%d.jpg" % yellow_count, yellow_img_crop)

        image = yellow_img_crop
        nx,ny,ch = image.shape
        image = np.reshape(image, (nx * ny, ch))
        yellow_to_train.append(image[k,:])

orange_to_train = np.array(orange_to_train)
green_to_train = np.array(green_to_train)
yellow_to_train = np.array(yellow_to_train)

np.save(path+'/Videos/Training_Frames/Orange_Frames/Train_orange.npy', orange_to_train)
np.save(path+'/Videos/Training_Frames/Green_Frames/Train_green.npy', green_to_train)
np.save(path+'/Videos/Training_Frames/Yellow_Frames/Train_yellow.npy', yellow_to_train)

orange_hist = []
green_hist = []
yellow_hist = []

for sample in range(len(orange_circle_train)):
    for row in range(len(orange_circle_train[sample])):
        for pixel in range(len(orange_circle_train[sample][row])):
            orange_hist.append(orange_circle_train[sample][row][pixel][0])

for sample in range(len(green_circle_train)):
    for row in range(len(green_circle_train[sample])):
        for pixel in range(len(green_circle_train[sample][row])):
            green_hist.append(green_circle_train[sample][row][pixel][1])

for sample in range(len(yellow_circle_train)):
    for row in range(len(yellow_circle_train[sample])):
        for pixel in range(len(yellow_circle_train[sample][row])):
            yellow_hist.append(yellow_circle_train[sample][row][pixel][0])
            yellow_hist.append(yellow_circle_train[sample][row][pixel][1])


#end Color Training
#######
#start Generate Training Histogram

confidence = 0.95
orange_int, mean_o, sigma_o = confidence_int(orange_hist,confidence)
green_int, mean_g, sigma_g = confidence_int(green_hist,confidence)
yellow_int, mean_y, sigma_y = confidence_int(yellow_hist,confidence)

print(orange_int)
print(green_int)
print(yellow_int)

outR = open("Videos/Training_Frames/Orange_Frames/Orange_Hist.txt", "w")
outG = open("Videos/Training_Frames/Green_Frames/Green_Hist.txt", "w")
outY = open("Videos/Training_Frames/Yellow_Frames/Yellow_Hist.txt", "w")

for r in orange_hist:
    outR.write(str(r))
    outR.write("\n")

for g in green_hist:
    outG.write(str(g))
    outG.write("\n")

for y in yellow_hist:
    outY.write(str(y))
    outY.write("\n")

outR.close()
outG.close()
outY.close()


orangeHist = plt.figure(1)
plt.hist(orange_hist, bins=256, range=(0,256), color="red")
plt.axvline(orange_int[0], color='k', linewidth=2)
plt.axvline(orange_int[1], color='k', linewidth=2)
plt.title("Red Histogram")
plt.ylabel('Occurrence')
plt.xlabel('Red Data')

GreenHist = plt.figure(2)
plt.hist(green_hist, bins=256, range=(0,256), color="green")
plt.axvline(green_int[0], color='k', linewidth=2)
plt.axvline(green_int[1], color='k', linewidth=2)
plt.title("Green Histogram")
plt.ylabel('Occurrence')
plt.xlabel('Green Data')

YellowHist = plt.figure(3)
plt.hist(yellow_hist, bins=256, range=(0,256), color="yellow")
plt.axvline(yellow_int[0], color='k', linewidth=2)
plt.axvline(yellow_int[1], color='k', linewidth=2)
plt.title("Yellow Histogram")
plt.ylabel('Occurrence')
plt.xlabel('Yellow Data')

plt.show()

train_check = 1

orange_gauss = plot_gauss(mean_o, sigma_o, 'r', "Red Gaussian Distribution")
green_gauss = plot_gauss(mean_g, sigma_g, 'g', "Green Gaussian Distribution")
yellow_gauss = plot_gauss(mean_y, sigma_y, 'y', "Yellow Gaussian Distribution")

fig, ax=plt.subplots()
ax.plot(orange_gauss,color='r')
ax.plot(green_gauss,color='g')
ax.plot(yellow_gauss,color='y')
plt.show()

print("Color training is complete")

#end Generate Training Histogram
#########
   
    
