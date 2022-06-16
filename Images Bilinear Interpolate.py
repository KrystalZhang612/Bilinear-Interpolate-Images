
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import numpy as np

from PIL import Image

import time

import imageio

#supplemental functions 

#starting by implementing convolution function focusing on kernel of the original image

def ConvolutionFunction(OriginalImage, Kernel):
    
    #the image.shape[0] of the original image represents its height 
    
    ImageHeight = OriginalImage.shape[0]
    
    #the image.shape[1] of the original image represents its width
    
    ImageWidth = OriginalImage.shape[1]
    
    #since kernel here represents a small 2d matrix to blur the original image 
    
    #We represents Kernel.shape[0] as its height, and Kernel.shape[1] as its width 
    
    KernelHeight = Kernel.shape[0]
    
    KernelWidth =  Kernel.shape[1]
    
    #pad numpy arrays within the image
    
    #we consider OriginalImage as an array 
    
    #if the grayscale image gives three element as the number of channels.
    
    
    if(len(OriginalImage.shape) == 3):
        
        PaddedImage = np.pad(OriginalImage, pad_width = ((KernelHeight // 2, KernelHeight // 2), 
        (KernelWidth//2, KernelWidth//2), (0,0)), mode='constant', constant_values=0).astype(np.float32)
        
        
        #if the grayscale image gives two element as the number of channels.
        
        
    elif (len(OriginalImage.shape) == 2):
        
        PaddedImage = np.pad(OriginalImage, pad_width = (( KernelHeight // 2,  KernelHeight // 2),
            (KernelWidth//2, KernelWidth//2)), mode='constant', constant_values=0).astype(np.float32)
        
        
    #floor division result quotient of Kernel height and width divides by 2 
        
    height = KernelHeight // 2
    
    width = KernelWidth // 2
    
    
    #initialize a new array of given shape and type, filled with zeros from padded image 
    
    ConvolvedImage = np.zeros(PaddedImage.shape)
    
    #sum = 0
    
    #iterate the image convolution as 2d array as well 
    
    for i in range(height, PaddedImage.shape[0] - height):
        
        for j in range(width, PaddedImage.shape[1] - width):
            
            
            #2D matrix indexes 
            
            x = PaddedImage[i - height:i-height + KernelHeight, j-width:j-width + KernelWidth]
            
            #use flaten() to return a copy of the array collapsed into one dimension.
            
            x = x.flatten() * Kernel.flatten()
            
            #pass the sum of the array elements into the convolved image matrix
            
            ConvolvedImage[i][j] = x.sum()
            
    #assign endpoints of height and width in the 2D matrix 
            
    HeightEndPoint = -height
    
    WidthEndPoint  = -width 
    
    #when there is no height, return [height, width = width end point] 
    
    if(height == 0):
        
        return ConvolvedImage[height:, width : WidthEndPoint]
    
    #when there is no width, return [height = height end point, width ] 
    
    if(width  == 0):
        
        return ConvolvedImage[height: HeightEndPoint, width:]
    
    #return the convolved image
    
    return ConvolvedImage[height: HeightEndPoint,  width: WidthEndPoint]


#implement a nearest function to get the nearest neighbor interpolation 
                  
def NearestInterpolation(image):
    
    #read image with imageio.imread(image)
    
    image = imageio.imread(image)
    
    #assign the 2D array of the old image 
    
    oldImage = np.asarray(image)
    
    #to be 4 times larger in each direction 

    factor = 4

    Width, Height, Column = image.shape
    
    newWidth  = int(Width*factor)
    
    newHeight = int(Height*factor)
    
    #get the width, height and column 
    
    newImage = np.zeros((oldImage.shape[0]*factor,oldImage.shape[1]*factor,oldImage.shape[2]), dtype=np.float32)
    

    for Row in range(newWidth):
        
        for Col in range(newHeight):
            
            newImage[Row, Col] = oldImage[Row//4, Col//4]
            

    return (newImage.astype(np.uint8))


#Driver code/testing: 

#Upsample image ”Moire small.jpg” to be 4 times larger in 
#each direction (16 times more image area) once with nearest neighbor interpolation and save as ”6a.png”

a = NearestInterpolation('Moire_small.jpg')

plt.imshow(a)

plt.imsave('6a.png', a)

#supplemental function for bilinear inerpolaition:

#similar implementation compared to nearest neighbor interpolation 

def BilinearFunction(image):
    
    image = imageio.imread(image)
    
    oldImage = np.asarray(image)
    
    factor = 4
    
    Width, Height, Column= image.shape
    

    newWidth = int(Width*factor)
    
    newHeight = int(Height*factor)
    
    newImage_neighbour = np.zeros((oldImage.shape[0]*factor,oldImage.shape[1]*factor,oldImage.shape[2]), dtype=np.float32)
    
    
    newImage = np.zeros((oldImage.shape[0]*factor,oldImage.shape[1]*factor,oldImage.shape[2]), dtype=np.float32)
    
    
    for k in range(image.shape[2]):
        
        for Row in range(newWidth):
            
            for Col in range(newHeight):
                
                #call the bilinear interpolation function here which will further be implemented 
                
                newImage[Row, Col, k], newImage_neighbour[Row, Col, k] = BilinearInterpolation(image[:,:,k], Row, Col) 
                
                
    
    return (newImage.astype(np.uint8))


#we also need a calculate function to multiply rows and columns 

def CalculateFunction(x, y, image):
    
    a = x/4.0 - x//4
    
    b = y/4.0 - y//4
    
    
    x = x//4
    
    y = y//4
    
    Width = (1-a)*(1-b)*image[x, y]
    
    if x+1 < image.shape[0]:
        
        Width = Width +(a *(1-b)*image[x+1, y])
        
    if y+1 < image.shape[1]:
        
        Width = Width + (1-a)*b* image[x, y+1]
        
    if x+1 < image.shape[0] and y+1 < image.shape[1]:
        
        Width = Width +(a*b*image[x+1, y+1])
        
    return Width 


def BilinearInterpolation(image, x, y):

    xx = x//4
    
    yy = y//4
    
    return CalculateFunction(x,y, image),image[xx][yy]


#Driver code for upsampling with bilinear interpolation

#Required test: once with bilinear interpolation and save as ”6b.png”.



b = BilinearFunction('Moire_small.jpg')

plt.imshow(b)

plt.imsave('6b.png',b)
