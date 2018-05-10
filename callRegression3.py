
# coding: utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import os
import tifffile
from numpy import amax
import thunder as td
import matplotlib.pyplot as plt
import registration 
os.umask(002)
import regression
from pyspark import SparkConf, SparkContext
import pandas as pd
sc = SparkContext() 
import osDB
import scipy.io
from PIL import Image
from PIL.ExifTags import TAGS
import pandas as pd
folderN = '_regression3' #name at the end of folder where files are placed

#load csv containing target paths
savepath = '/groups/flyfuncconn/home/busheyd/scripts/Python/regressionStim/currentRegression3.csv'
paths = pd.read_csv(savepath)

#find the first path that has not been started
indx = numpy.where(paths['Progress'].values == 'None')
cpath = paths['Paths'].iloc[indx[0][0]]

## load images
path = os.path.join(cpath, '')
#rawdata = td.images.fromtif(path)
rawdata = td.images.fromtif(path, engine=sc)
stacks = rawdata.count()


## get the file (ending in stim.mat) that has the times stimulated
filename, indx = osDB.getFileContString(path, 'stim.mat')
stim_times_file = path  + filename.values[0]
mat = scipy.io.loadmat(stim_times_file)
times = numpy.where(numpy.diff(numpy.squeeze(mat['AOBuffer'])) > 0)
times = times[0] /100 #convert to seconds

#get the actual time for each tif
#function to find the tag
def findPilTag(imgPil, tagStr):
    #imgPil = loaded Pil image cimg = Image.open(path + '/' + cfile)
    #tagStr = 'string' to search for in the tags
    for key1 in imgPil.tag:
        ctag= imgPil.tag[key1][0]
        
        if isinstance(ctag, unicode):
            index = ctag.find(tagStr)
            timestamp = ctag[index +len(tagStr) : ]
            timestamp = timestamp.split()
            timestamp = timestamp[0]
            #import pdb; pdb.set_trace()

    return timestamp

#get list of tif files
filename, indx = osDB.getFileContString(path, '.tif')
lastTifFile = filename.sort_values().values

#now get the numbers
times1 = []
for cfile in lastTifFile:
    cimg = Image.open(path + '/' + cfile)
    times1.append(float(findPilTag(cimg, 'frameTimestamps_sec =')))
    
## find the number stacks start before the corresponding to stimulus times
times2 = []
for ctime in times:
    times2.append( numpy.where(numpy.array(times1) <= ctime)[0][-1])
'''
I don't believe times are being set correctly consider revising
index1 = np.where(stimTimes>0)
binTimes = np.zeros(len(stimTimes))
binTimes[index1[0]] = 1
start = np.where(np.diff(binTimes)==1)[0]+1
stop = np.where(np.diff(binTimes)==-1)[0]+1
index2 = np.zeros(len(stimTimes))
frametimes =  np.arange(rawshape[3]) * framerate
StimFramesIndex = np.zeros(rawshape[3])
for i, cstart in enumerate(start):
    start1 = np.where(frametimes >= cstart/100)[0][0]
    stop1 = np.where(frametimes <= stop[i]/100)[0][-1]
    StimFramesIndex[start1:stop1+1] = 1
print(np.where(StimFramesIndex ==1))
see A28_GroupByCrossC.ipynb
'''
    
## write stimulus times
detect ='fromfile'

#get filename
path_base, path_head = os.path.split(path)
path_base, flyID =os.path.split(path_base)
#make dir
if not os.path.exists(path_base+os.path.sep+flyID+folderN+os.path.sep):
    os.makedirs(path_base+os.path.sep+flyID+folderN+os.path.sep)    

#save stim time data
stimData = pd.DataFrame({'Stimulation_time': times, 'Stack_#': times2, 'Frame_Time': numpy.array(times1)[times2], 'Next_Frame_Time': numpy.array(times1)[numpy.array(times2)+1] } )
#times = times2
stimData=stimData[['Stimulation_time', 'Stack_#', 'Frame_Time', 'Next_Frame_Time']]
fname = path_base+os.path.sep+flyID+folderN+os.path.sep+flyID+'_stimtimes'+detect+'.csv'  
stimData.to_csv(fname)
times = times2


# In[5]:

## registration -- correct for movement in image
fname = path_base+os.path.sep+flyID+folderN+os.path.sep+flyID+'_registration.json'
#create new registration
reg = registration.CrossCorr()
reference = rawdata[0:30, : , :, :].mean().toarray()
#reference = rawdata.mean().toarray()
#plt.imshow(amax(reference,0), cmap='gray', clim=(0,2000))
#reg.prepare(rawdata, startIdx=100, stopIdx=110);
registrationModel = reg.fit(rawdata, reference = reference)
#registrationModel.save(fname, overwrite=True)
#import json
#with open(fname, 'w') as outfile:
#    json.dump(registrationModel, fname)


# In[6]:

displacements = registrationModel.toarray()


# In[7]:

images = registrationModel.transform(rawdata)
images.cache()


# In[8]:

# save z-projections as tiff stack
# put this before coalesce.  coalesce messes up images, not sure why
#mipImg = images.maxProjection(axis=2).collectValuesAsArray()
mipImg = images.max_projection(axis=0).toarray()
#save file
fname = path_base+os.path.sep+flyID+folderN+os.path.sep+flyID+'_MIP.tif' 
tifffile.imsave(fname, mipImg.astype(numpy.uint16))
#tifffile assumes data to be image depth, height, width, and samples.


# In[9]:

## convert the data to binary series format (essentially a matrix of voxels by time)
## only need to run once so it's commented out
#data.saveAsBinarySeries(path + 'binary', overwrite=True)
#data = tsc.loadSeries(path + 'binary').normalize('mean')
#data = tsc.loadSeries(path + 'binary')
# alternate version
#data = images.medianFilter(3).toTimeSeries().normalize('mean')

#td.series.Series.aggregate_by_index
#data1 = images.median_filter(3)
data = images.median_filter(3).toseries().normalize('mean')
#data = images.median_filter(3).dottimes().normalize('mean')
#data = images.toTimeSeries().normalize('mean')
data.cache()
data.compute()


# In[10]:

events=numpy.zeros(stacks)
events[times]=1


# In[11]:

# build a design matrix with shifted copies of the impulse function
from numpy import roll, zeros
maxshift = 20

num_stacks = numpy.size(events)
designmatrix = zeros((maxshift,num_stacks))
for i in range(0,maxshift):
    designmatrix[i,:] = roll(events,i)
#image(designmatrix) #just plotting function



# In[12]:

## construct the model
#print dir(regression)
model = regression.LinearRegression(fit_intercept = False)


# In[13]:

# ## before fitting to all voxels, we'll fit to the mean
# ## there's a decent signal in the mean,
# ## and this will be an easy way to see what the regression is doing
# y = data.mean()

# ## apply the model to this one signal
# from thunder.regression.estimators import PseudoInv
# estimator = PseudoInv(designmatrix.T)
# b_example = estimator.estimate(y)
# plt.plot(b_example)


# In[14]:

# ## show how the regression coefficients are computed
# ## (this happens inside model.get),
# ## and show the prediction (red) versus the real response (blue)
# ## note the strong nonlinearity that we're not capturing with a simple linear model
# from numpy import dot
# b = dot(estimator.Xhat, y)
# predic = dot(b, designmatrix)
# plt.plot(y)
# plt.plot(predic, color='r')


# In[15]:

fitted = model.fit(designmatrix.T, data)


# In[16]:

results = fitted.score(designmatrix.T, data)


# In[17]:

results=numpy.squeeze(results.toarray())


# In[18]:

## save R2 statistics as tiff stack

#save file
fname = path_base+os.path.sep+flyID+folderN+os.path.sep+flyID+'_reg2.tif'
#fname = path_base+os.path.sep+flyID+folderN+os.path.sep+flyID+'_10newstim.tif'
#print fname

#temp = numpy.swapaxes(results, 0, 1)
#temp = numpy.swapaxes(temp, 0, 2)
tifffile.imsave(fname, (results*2**16).astype(numpy.uint16))
## tifffile assumes data to be image depth, height, width, and samples.


# In[19]:

## save baseline image that averages all imaging frames as tiif stack
meanImg = images.coalesce(10).mean()  

#save file
fname = path_base+os.path.sep+flyID+folderN+os.path.sep+flyID+'_baseline.tif'

tifffile.imsave(fname, meanImg.squeeze().astype(numpy.uint16).toarray())
## tifffile assumes data to be image depth, height, width, and samples.


# In[20]:

fig1=plt.figure(figsize=(10,8), dpi = 600)
ax1 = fig1.add_axes([0.01, 0.3, 0.4, 0.4])
plt.axis('off')
plt.imshow(amax(images.first(),0), cmap='gray', clim=(0,numpy.max(images.first(),)))
plt.title('Maximum Image Projection First Stack')
ax2 = fig1.add_axes([0.01, 0.01, 0.4, 0.4])
plt.imshow(designmatrix)
plt.axis('off')
plt.title('Design Matrix with shifted copies of the impulse function')
ax3 = fig1.add_axes([0.5, 0.01, 0.4, 0.9])
plt.imshow(amax(results.clip(0,1),0), cmap='gray')
plt.axis('off')
plt.title('MIP: Regression Analysis')
plt.savefig(path_base+os.path.sep+flyID+folderN+os.path.sep+flyID+'_Summary.jpeg')

## update csv file after finishing regression
paths['Progress'].iloc[indx[0][0]] = 'Finished'
paths.to_csv(savepath, index = False)





