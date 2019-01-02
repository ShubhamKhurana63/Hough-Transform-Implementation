import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

image=cv2.imread("C://Users/Shubham/Documents/UB/CVIP/project3/original_imgs/hough.jpg",0)
def findMax(arr):
    max=0;
    for i in range(0,len(arr)-1):
        for j in range(0,len(arr)-1):
              if arr[i][j] > max:
                    max=arr[i][j];
    return max;


def padImage(image):
    return cv2.copyMakeBorder( image, 1,1, 1, 1, cv2.BORDER_CONSTANT,None,0)

#flipping before the convolution
def flipSobel(kernel):
    for i in range(0,3):
        temp=kernel[0][i]
        kernel[0][i]=kernel[2][i];
        kernel[2][i]=temp;
    for i in range(0,3):
        temp=kernel[i][0];
        kernel[i][0]=kernel[i][2];
        kernel[i][2]=temp;
    print(kernel)
    return kernel;
#for displaying the image
def showImage(edge):
    cv2.imshow('image', edge);
    cv2.waitKey(0);
    cv2.destroyAllWindows();

 #for padding the image, for detecting the edges at the corner   
# def paddingImage(img):
#     list=[[ 0 for x in range(0,img.shape()+2)] for y in range(0,img.shape()+2)];
#     paddedImage=np.asarray(list)
#     paddedImage[1:601,1:901]=img;
#     return paddedImage;    
edgeList=[];
#applying the sobel operator on the image
def performConvolutionForSobel(img,sobel_template):
    sumList=[];
    paddedImage=padImage(img);
    sobel_template=flipSobel(sobel_template)
    list=[[ 0.0 for x in range(0,img.shape[1])] for y in range(0,img.shape[0])];
    my_arr=np.asarray(list);
    for row in range(1,paddedImage.shape[0]-1):
        for column in range(1,paddedImage.shape[1]-1):
            sum=0;
            for i in range(-1,2):
                for j in range(-1,2):
                    sum=sum+paddedImage[row+i,column+j] * sobel_template[i+1,j+1];
                my_arr[row-1,column-1]=np.absolute(sum);
            sumList.append(sum);
    edge=my_arr;
    edgeList.append(edge);
    maxPixel=max(sumList);
    #edge=edge/maxPixel;
    return edge;


img=cv2.imread("C://Users/Shubham/Documents/UB/CVIP/project3/original_imgs/hough.jpg",0)
#newImageArr=cv2.imread("C://Users/Shubham/Documents/UB/CVIP/proj1_cse573/task1.png",0);
#paddedImage=padImage(img);
sobel_x_template=np.asarray([[-1,0,1],[-2,0,2],[-1,0,1]]);
sobel_y_template=np.asarray([[1,2,1],[0,0,0],[-1,-2,-1]]);


edge_x=performConvolutionForSobel(img,sobel_x_template)
#showImage(edge_x)
edge_y=performConvolutionForSobel(img,sobel_y_template)
#showImage(edge_y)
edge_magnitude=(edge_x**2+edge_y**2)**(1/2)

edge_magnitude=edge_magnitude/np.max(edge_magnitude)
edge_magnitude=edge_magnitude*255
#showImage(edge_magnitude)
ret,edge_magnitude=cv2.threshold(edge_magnitude,45,255,cv2.THRESH_BINARY)



#################################################################################################################3


coloredImage=cv2.imread("C://Users/Shubham/Documents/UB/CVIP/project3/original_imgs/hough.jpg",1)
#detectionKernel=np.asarray([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
#binaryImage=image/255
FILE_SAVING_PATH="C://Users/Shubham/Documents/UB/CVIP/project3/code/result_images/"

def showImage(edge):
    cv2.imshow('image', edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getEdgedimage(image):
    #250,300
    edgeImage=cv2.Canny(image,250,300)
    return edgeImage


def findDiagonalLength(image):
    return (image.shape[0]**2+image.shape[1]**2)**(0.5)

#edgeImage=getEdgedimage(image)
#showImage(edge_magnitude)
#ret,thresh1 = cv2.threshold(edge_magnitude,70,255,cv2.THRESH_BINARY)

for i in range(0,edge_magnitude.shape[0]):
    for j in range(0,edge_magnitude.shape[1]):
        if edge_magnitude[i][j]>70:
            edge_magnitude[i][j]=255
        else:
            edge_magnitude[i][j]=0

#assigning the sobel edge detecetd image
edgeImage=edge_magnitude

#computing diagonal length
diagonalLength=int(findDiagonalLength(image))

#creating p-theta array(1638*181)
pThetaArray=np.zeros((diagonalLength*2,181),dtype=np.float32)
print('------',pThetaArray.shape)
offset=int(pThetaArray.shape[0]/2)

#normal line equation in polar form
def fillPThetaSpace(x,y,pThetaArray):
    for theta in range(-90,91,1):
        p=x*math.cos(np.deg2rad(theta))+y*math.sin(np.deg2rad(theta))
        pThetaArray[offset+int(p)][theta+90]=pThetaArray[offset+int(p)][theta+90]+1
    


def houghAlgo(image,pThetaArray):
    Y,X=np.nonzero(image)
    for i,j in zip(Y,X) :
        fillPThetaSpace(j,i,pThetaArray)
    #return pThetaArray
houghAlgo(edgeImage,pThetaArray)

print('--------------done--------------')

#filtering lines, collecting the lines for a particular angle and over a particular line in one bucket in map
def filterLines(thetaList,pArray,thetaArray,dFactor):
    listLineMap={}
    redundantCheck=[]   
    for i,j in zip(pArray,thetaArray):
        if thetaList.count(j)==1 and redundantCheck.count(i)==0:
            index=math.floor(i/dFactor)
            redundantCheck.append(i)
            if listLineMap.get(index) is not None:
                subList=listLineMap.get(index)               
                subList.append((i,j))
                listLineMap[index]=subList
            else:
                nList=[]
                nList.append((i,j))
                listLineMap[index]=nList
    return listLineMap

#drawing lines(computing coordinates and then plotting lines)
def drawLines(dFactor,thetaList):
    mapOfLines=filterLines(thetaList,pArray,thetaArray,dFactor)
    #mapOfLines=filterThetaForRed(mapOfLines)
    #print(mapOfLines.values())
    ctr=0
    for z in mapOfLines.values():
        if len(z)>2:
            element=z[math.floor((len(z))/2)]
        else:
            element=z[len(z)-1]
        p=element[0]
        theta=element[1]
        print(p,'-----',theta)
        a=math.cos(np.deg2rad(theta))
        b=math.sin(np.deg2rad(theta))
        x0=p*a
        y0=p*b
        x1=int(x0+800*(-b))
        y1=int(y0+800*(a))
        x2=int(x0-800*(-b))
        y2=int(y0-800*(a))
        cv2.line(coloredImage,(x1,y1),(x2,y2),(0,0,0),2)
        ctr=ctr+1
    return coloredImage





def writeImage(image,name):
    cv2.imwrite(FILE_SAVING_PATH+name+'.jpg',image)    


#red line
coloredImage=cv2.imread("C://Users/Shubham/Documents/UB/CVIP/project3/original_imgs/hough.jpg",1)
coordinates=np.where((pThetaArray>90) )
pArray=coordinates[0]-819
thetaArray=-90+coordinates[1]
coloredImage=drawLines(100,[-2])
showImage(coloredImage)
writeImage(coloredImage,'redline')


#blue lines
# coloredImage=cv2.imread("C://Users/Shubham/Documents/UB/CVIP/project3/original_imgs/hough.jpg",1)
# coordinates=np.where((pThetaArray>140))
# pArray=coordinates[0]-819
# thetaArray=-90+coordinates[1]
# #coloredImage=drawLines(80,[-36])
# drawLinesP(pArray,thetaArray,coloredImage)
# showImage(coloredImage)


#blue lines
coloredImage=cv2.imread("C://Users/Shubham/Documents/UB/CVIP/project3/original_imgs/hough.jpg",1)
coordinates=np.where((pThetaArray>120) )
pArray=coordinates[0]-819
thetaArray=-90+coordinates[1]
coloredImage=drawLines(65,[-36])
showImage(coloredImage)
writeImage(coloredImage,'blueline')

print('---------max value--------',np.max(pThetaArray))



#############################################BONUS TASK#########################################################


#radius=23
coloredImage=cv2.imread("C://Users/Shubham/Documents/UB/CVIP/project3/original_imgs/hough.jpg",1)   
rowsForCircle=edgeImage.shape[0]
columnsForCircle=edgeImage.shape[1]
def fillCircleThetaSpace(x,y,pThetaArray,radius,radiusCounter):
    for theta in range(0,361,1):
        b=y-np.around(radius*np.sin(np.deg2rad(theta)))
        a=x-np.around(radius*np.cos(np.deg2rad(theta)))
        if a>=0 and a <rowsForCircle and b>=0 and b<columnsForCircle:
            pThetaArray[int(a)][int(b)][radiusCounter]=pThetaArray[int(a)][int(b)][radiusCounter]+1
          
radiusList=[21,22,23]
def houghAlgoForCircle(image,pThetaArray):
    X,Y=np.nonzero(image)
    ctr=0
    for radius in  radiusList:
        for i,j in zip(X,Y):
            fillCircleThetaSpace(i,j,pThetaArray,radius,ctr)
        ctr=ctr+1

def drawCircles(circleIndices):
    circleVertices=[]
    for i,j,k in zip(circleIndices[0],circleIndices[1],circleIndices[2]):
        circleVertices.append((i,j,radiusList[k]))
    for z in circleVertices:
        cv2.circle(coloredImage,(z[1],z[0]), z[2], (255,0,0), 2)


pThetaArrayCircle=np.zeros((edgeImage.shape[0],edgeImage.shape[1],len(radiusList)))
houghAlgoForCircle(edgeImage,pThetaArrayCircle)
circleIndices=np.where(pThetaArrayCircle>np.max(pThetaArrayCircle)*0.529)
drawCircles(circleIndices)
showImage(coloredImage) 
writeImage(coloredImage,'coin')
