
from re import S
import cv2
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

#using function from someone else in here: (Cited inside the transform script)
from transform import HSI2RGB,RGB2HSI


#%% function to read images

def read_images(images_dir):
    
    #create directories of images and masks
    
    input_img_paths = sorted(
        [
        os.path.join(images_dir, fname) #join the path and the name where name is given by next line
        for fname in os.listdir(images_dir) #name each file in the directory
        if fname.endswith(".jpg") or fname.endswith('.jpeg') or fname.endswith('.png') #aceptable formats for files in directory
        ]
    )    

    x=[]

    for i in range(0,len(input_img_paths)):

        #load images as multiple numpys arrays
        I = cv2.cvtColor(cv2.imread(input_img_paths[i]), cv2.COLOR_BGR2RGB)
    
        I3= np.array(I)
        x.append(I3)
      
    #transform data into numpy arrays
    x=np.array(x)

    return x

input_dir = "./Dataset/test/hazy" #image to enhance

# Read images
x = read_images(input_dir)

print("input_images shape and range", x.shape, x.min(), x.max())


#take one example image to test the algorithm
I_original = x[1,:,:,:]

#code for plotting purposes from: https://datacarpentry.org/image-processing/05-creating-histograms/ 

def plot_histo(ima):
    # tuple to select colors of each channel line
    colors = ("red", "green", "blue")
    channel_ids = (0, 1, 2)

    # create the histogram plot, with three lines, one for
    # each color
    plt.figure()
    plt.xlim([0, 256])
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(
            ima[:, :, channel_id], bins=256, range=(0, 256)
        )
        plt.plot(bin_edges[0:-1], histogram, color=c)

    plt.title("Color Histogram")
    plt.xlabel("Color value")
    plt.ylabel("Pixel count")

    plt.show()


#plot the original image histogram
plot_histo(I_original)


#%% 1. First strech the color channels

def strech_color(ima):
    #input ima is an rgb image [hight,wide,num_colors]
    [a,b,c]=ima.shape

    final_ima=np.zeros((a,b,c)) #the return with already streched channels

    for channel in range(0,c): # iterates over each channel

        c_min=np.min(ima[:,:,channel])
        c_max=np.max(ima[:,:,channel])

        for i in range(0,a): #iterates over each row
            for j in range(0,b): #iterates over each column
                final_ima[i,j,channel] = (ima[i,j,channel] - c_min)/(c_max-c_min)

    return(final_ima)

p1=strech_color(I_original)
print("Streched image: ", p1.shape, np.min(p1), np.max(p1))

#%% 2. Get the intesity of each channel by using RGB to HSI and find value of I (intensity of each pixel matrix)

def HSI(ima):
    #input ima is an rgb image [hight,wide,num_colors]
    [a,b,c]=ima.shape

    I=np.zeros((a,b)) #the return of intesty component of the HSI color space transformation

    for i in range(0,a): #iterates over each row
        for j in range(0,b): #iterates over each column
            I[i,j] = (((ima[i,j,0]+ima[i,j,1]+ima[i,j,2])/3))

    return(I)

p2=HSI(p1) #get the intesity pixel matrix
print("Intensity matrix: ", p2.shape, np.min(p2), np.max(p2)) #it distributes between 0 and 1
#print(p2)

#%% Now compute the dissimilarity histogram

#We start by computting the membership function

def MF(I,std_ima): #membership function
    #input the intensity matrix and the standar deviation of the original image

    [a,b]=I.shape

    pmf=np.zeros((a,b)) #to create a similar fuzy set
    u_d=np.zeros((a,b)) #create a similar fuzy set

    for i in range(0,a): #rows
        for j in range(0,b): #columns

            u_temp=[] #to save the tmeporary calculation of U values before taking maximum

            #we are going to try the 3x3 operations (9 in total). But use try-except in case we are in a border pixel
            
            #left pixels
            try:
                r1=abs(1-I[i,j]-I[i-1,j])/std_ima
                if r1>0:
                    u_temp.append(r1)
                else:
                    u_temp.append(0)
            except:
                pass

            try:
                r2=abs(1-I[i,j]-I[i-1,j-1])/std_ima
                if r2>0:
                    u_temp.append(r2)
                else:
                    u_temp.append(0)
            except:
                pass

            try:
                r3=abs(1-I[i,j]-I[i-1,j+1])/std_ima
                if r3>0:
                    u_temp.append(r3)
                else:
                    u_temp.append(0)
            except:
                pass

            #upper an lower pixels
            try:
                r4=abs(1-I[i,j]-I[i,j-1])/std_ima
                if r4>0:
                    u_temp.append(r4)
                else:
                    u_temp.append(0)
            except:
                pass

            try:
                r5=abs(1-I[i,j]-I[i,j+1])/std_ima
                if r5>0:
                    u_temp.append(r5)
                else:
                    u_temp.append(0)
            except:
                pass

            #right pixels
            try:
                r6=abs(1-I[i,j]-I[i+1,j])/std_ima
                if r6>0:
                    u_temp.append(r6)
                else:
                    u_temp.append(0)
            except:
                pass

            try:
                r7=abs(1-I[i,j]-I[i+1,j+1])/std_ima
                if r7>0:
                    u_temp.append(r7)
                else:
                    u_temp.append(0)
            except:
                pass

            try:
                r8=abs(1-I[i,j]-I[i+1,j-1])/std_ima
                if r8>0:
                    u_temp.append(r8)
                else:
                    u_temp.append(0)
            except:
                pass

            #create a similar fuzzy set
            pmf[i,j]=np.mean(u_temp)

            #finally we take the complement of pmf
            u_d[i,j]=(1-pmf[i,j])
    
    return(u_d)

p3=MF(p2,np.std(I_original)) #get new "dissimilar" fuzzy set --> standar deviation of the original image
print("MF: ",p3.shape,np.min(p3),np.max(p3))

#%% 4. compute the fuzzy dissimilarity histogram Hfd

def fuzzy_histo(intensity, dissimilar): #get the fuzzy histogram
    #input the dissimilarity matrix and the intensiy matrix for campare them
    [a,b]=intensity.shape

    print("i mat: ",intensity)
    print("d mat: ",dissimilar)

    #miu_pi=np.zeros((a,b))
    histo=np.zeros((1,256))

    for k in range(0,256):
        
        #miu_pi=np.zeros((a,b))
        miu_pi=0

        for i in range(0,a):#rows
            for j in range(0,b): #colum

                if int(intensity[i,j])==k:
                    miu_pi=miu_pi+dissimilar[i,j]
        histo[0,k]=miu_pi

    return histo


#transform p2 in range of values 0 to 255
def trans_255(intensity):
    [t1,t2]=intensity.shape
    r=np.zeros((t1,t2))
    for i in range(0,t1):
        for j in range(0,t2):
            r[i,j]=int((int(intensity[i,j]*1000)/1000)*255)
    return r

p2_255=trans_255(p2)


p4=fuzzy_histo(p2_255, p3)
print("fuzzy histogram: ",p4.shape,np.min(p4),np.max(p4))


#plot historgram
plt.figure()
[a,b]=(p4).shape
plt.bar(np.arange(b),p4[0,:])
plt.title("Fuzzy Histogram")
plt.xlabel("Intensity")
plt.ylabel("Value")
plt.show()


#%% 5. gamma correction

#pdf
def pdf(histo):
    [a,b]=histo.shape

    pdf=np.zeros((a,b))

    total_pixels=np.sum(histo[0,:])

    for i in range(0,b):
        pdf[0,i]=histo[0,i]/total_pixels
    
    return pdf

p5=pdf(p4) #normalize histogram between 0 and 1 in axis-y
print("pdf: ",p5.shape,np.min(p5),np.max(p5))


#plot historgram
plt.figure()
[a,b]=(p5).shape
plt.bar(np.arange(b),p5[0,:])
plt.title("PDF")
plt.xlabel("Intensity")
plt.ylabel("Value")
plt.show()


#compute the comulative density function
def CDM(histo):
    [a,b]=histo.shape

    cdm=np.zeros((a,b))
    
    for k in range(0,b):
        if k==0: #if its the first intensity
           cdm[0,k]=histo[0,k]
        else:
            cdm[0,k]= cdm[0,k-1] + histo[0,k] #add the previous and the k of histo

    return cdm

p6=CDM(p5) #cumulative histogram
print("cdm: ",p6.shape,np.min(p6),np.max(p6))


#plot historgram
plt.figure()
[a,b]=(p6).shape
plt.bar(np.arange(b),p6[0,:])
plt.title("CDM")
plt.xlabel("Intensity")
plt.ylabel("Value")
plt.show()


#%% 6. compute the weighted histogram distribution function pdf_W (WHD)

def WHD(pdf,cdf):

    [a,b]=pdf.shape

    pdf_w=np.zeros((a,b))

    pdf_max=np.max(pdf)
    pdf_min=np.min(pdf)

    for i in range(0,b):
        pdf_w[0,i]=pdf_max* (((pdf[0,i]-pdf_min)/(pdf_max-pdf_min))**cdf[0,i])
    
    return pdf_w

p7=WHD(p5,p6) #weighted histogram distribution function
print("pdf_w: ",p7.shape,np.min(p7),np.max(p7))


#plot historgram
plt.figure()
[a,b]=(p7).shape
plt.bar(np.arange(b),p7[0,:])
plt.title("PDF_W")
plt.xlabel("Intensity")
plt.ylabel("Value")
plt.show()


#%% 7. compute the weighted comulative distribution fuction

def CDFW(pdf_w):

    [a,b]=pdf_w.shape

    cdf_w=np.zeros((a,b))

    s=np.sum(pdf_w[0,:]) #sum of all weighted intensities

    p_sum=pdf_w/s

    for i in range(0,b):
        if i==0: #if is the first iteration
            cdf_w[0,i]=p_sum[0,i]
        else:
            cdf_w[0,i]=cdf_w[0,i-1] + p_sum[0,i] #previous plus the new
    
    return cdf_w

p8=CDFW(p7) #weighted histogram distribution function
print("cdf_w: ",p8.shape,np.min(p8),np.max(p8))


#plot historgram
plt.figure()
[a,b]=(p8).shape
plt.bar(np.arange(b),p8[0,:])
plt.title("CDM_W")
plt.xlabel("Intensity")
plt.ylabel("Value")
plt.show()


#%% 8. Compute gamma 

gamma=1-p8
print("gamma: ",gamma.shape,np.min(gamma),np.max(gamma))
#print(gamma)

#%% 9. apply gamma correction to intensity of streched colors

def apply_gamma(gamma,intensity):
    [a,b]=intensity.shape

    I_corrected=np.zeros((a,b))
    max_inte = np.max(intensity)

    for i in range(0,a):
        for j in range(0,b):
            val_temp=int(intensity[i,j])
            #I_corrected[i,j]=(intensity[i,j]*((intensity[i,j]/max_inte)**gamma[0,(val_temp)]))
            I_corrected[i,j]=(val_temp*((val_temp/max_inte)**gamma[0,(val_temp)]))

    return I_corrected

p9=apply_gamma(gamma,p2_255)
print("corrected intensity: ",p9.shape,np.min(p9),np.max(p9))

#%% trasform with corrected intensity back to RGB and then back to HSI to get the maximum saturation S

img_r = cv2.normalize(I_original, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) #resized image between 0 and 1, which is bassically x/255
#using dtype=cv2.CV_32F flag to convert from unit8 to float to make image show correctly

HSI_tempo1= RGB2HSI(img_r)
print('HS1_dimensions: ', HSI_tempo1.shape)

ima_c= np.dstack((HSI_tempo1[:,:,0],HSI_tempo1[:,:,1],p9))#corrected image with intensity
RGB_tempo = HSI2RGB(ima_c) #transform to rgb with corrected intensity matrix

#transform back to HSI
HSI_tempo2= RGB2HSI(RGB_tempo)

S_e = HSI_tempo2[:,:,1] #extracted from the gamma corrected intensity image
S_o = HSI_tempo1[:,:,1] #extracted from the original image

#%% get the final saturation

def final_s(S_e,S_o):

    [a,b]=S_e.shape

    S_m=np.zeros((a,b))

    for i in range(0,a):
        for j in range(0,b):
            if S_e[i,j] >= S_o[i,j]:
                S_m[i,j]=S_e[i,j]
            else:
                S_m[i,j]=S_o[i,j]

    return S_m

S_m=final_s(S_e,S_o)

#get the final rgb image

ima_c2= np.dstack((HSI_tempo2[:,:,0],S_m,p9))#corrected image with intensity and saturation
final_image = HSI2RGB(ima_c2)

#save image
save_dir = "./Dataset/results/" #image to enhance

img_br_save = cv2.normalize(final_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) #resized image between 0 and 255 to correctly save it
cv2.imwrite((save_dir+'10.png'),cv2.cvtColor(img_br_save, cv2.COLOR_RGB2BGR)) #save

#plot the corrected image histogram
plot_histo(img_br_save)

'''
plt.figure()
plt.imshow(final_image)
'''

