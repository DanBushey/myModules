# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:06:14 2016

@author: busheyd
"""
import numpy as np
import osDB
import subprocess
import tifffile
#import nibabel
import os
import code
#import nrrd
import pandas as pd
import PIL
import scipy.ndimage as nd
import nibabel
import time
import registration
#import path
import numpy as np
import registration
import skimage.filters as skfilters
import skimage.morphology as skmorphology
import scipy

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)
    
def runBigWarpTransform(srcFfile, itvlF, newCSVfile, newdirfile, affineFfile = ['None'], defFfile = ['None']):
    #code.interact(local=locals())
    #runs the java script runBigWarpTransform
    #srceFfile = baseline image  (image to be warped)
    #itvlF = nc82 Fru template image 
    #newCSVfile = csv landmark file from bigwarp 
    #newdirfile = directory to write transformed image
    #affineF = The affine part of the estimated transform in two different formats  I use mat 
    #def = <subject-prefix>_spacingWarp.nii.gz - The deformable part of the transform (this is the 4D image that represents a vector field) 
    #return srcFfile, itvlF, newCSVfile, newdirfile, affineFfile, defFfile
    #import pdb; pdb.set_trace()
    if os.path.isfile(srcFfile[0]) and os.path.isfile(newCSVfile[0]) and os.path.isfile(affineFfile[0])  and os.path.isfile(defFfile[0]):
        cmdL="/groups/flyfuncconn/home/busheyd/scripts/Python/Modules/runBigWarpTransform6"
        subprocess.check_call([cmdL, srcFfile[0], itvlF[0], newCSVfile[0], newdirfile[0], affineFfile[0], defFfile[0]])
    #subprocess.check_call([cmdL, srcFfile, itvlF, tpsFfile, newdirfile])
        output= ['Affine and Warp']
    else:
        cmdL="/groups/flyfuncconn/home/busheyd/scripts/Python/Modules/runBigWarpTransform4"
        subprocess.check_call([cmdL, srcFfile[0], itvlF[0], newCSVfile[0], newdirfile[0]])
        output=['landmark only']
    return [srcFfile, output]
      

def runantsRegistrationSynQuick(fixedImage, movingImage, outputimage):
    cmd = '/groups/flyfuncconn/home/busheyd/antsbin/bin/antsRegistrationSyNQuick.sh'
    #nameoption = 'PyReg'
    #targetfolder = movingImage[0 : movingImage.rfind('/')+1]
    #cmdL = cmd + ' -d 3 -f ' +fixedImage + ' -m ' + movingImage + ' -o ' + targetfolder + nameoption 
    cmdL = cmd + ' -d 3 -f ' +fixedImage[0] + ' -m ' + movingImage[0] + ' -o ' + outputimage[0]
    print cmdL
    output = subprocess.check_call(cmdL, shell = True)
    #return [output]
    return [cmdL]
    '''
    used to test function
    fixedImage  = '/groups/flyfuncconn/home/busheyd/scripts/Ants-Scripts/test/MYtemplate_uint8_baseline.nii.gz'
    movingImage = '/groups/flyfuncconn/home/busheyd/scripts/Ants-Scripts/test/20160301_r3_ol0020blc10_frup65_flya_00001_baseline-DB8.nii.gz'
    imageDB.runantsRegistrationSynQuick(fixedImage, movingImage)
   
   '''

def runantsRegistrationSynQuickGetCommand(fixedImage, movingImage, outputimage):
    cmd = '/groups/flyfuncconn/home/busheyd/antsbin/bin/antsRegistrationSyNQuick.sh'
    #nameoption = 'PyReg'
    #targetfolder = movingImage[0 : movingImage.rfind('/')+1]
    #cmdL = cmd + ' -d 3 -f ' +fixedImage + ' -m ' + movingImage + ' -o ' + targetfolder + nameoption 
    cmdL = cmd + ' -d 3 -f ' +fixedImage[0] + ' -m ' + movingImage[0] + ' -o ' + outputimage[0]
    print cmdL
    #output = subprocess.check_call(cmdL, shell = True)
    #return [output]
    return [cmdL]
    '''
    used to test function
    fixedImage  = '/groups/flyfuncconn/home/busheyd/scripts/Ants-Scripts/test/MYtemplate_uint8_baseline.nii.gz'
    movingImage = '/groups/flyfuncconn/home/busheyd/scripts/Ants-Scripts/test/20160301_r3_ol0020blc10_frup65_flya_00001_baseline-DB8.nii.gz'
    imageDB.runantsRegistrationSynQuick(fixedImage, movingImage)
    '''
   
def runantsRegistration(fixedImage, movingImage, outputFolder, outputWarp, outputInverseWarp):
    fixedImage=fixedImage[0]
    movingImage=movingImage[0]
    outputFolder = outputFolder[0]
    outputWarp = outputWarp[0]
    outputInverseWarp = outputInverseWarp[0]
    cmd = '/groups/flyfuncconn/home/busheyd/antsbin/bin//antsRegistration --verbose 1 --dimensionality 3 --float 0 --output '
    cmd = cmd + '[' + outputFolder + ',' + outputWarp + ',' + outputInverseWarp+ '] '
    cmd = cmd + '--interpolation Linear '
    cmd = cmd + '--use-histogram-matching 0 '
    cmd = cmd + '--winsorize-image-intensities [0.005, 0.95] '
    cmd = cmd + '--initial-moving-transform '
    cmd = cmd + '[' + fixedImage + ',' + movingImage + ',' + '1' + '] '
    cmd = cmd + '--transform Rigid[0.1] '
    cmd = cmd + '--metric MI'
    cmd = cmd + '[' + fixedImage + ',' + movingImage + ',' + '1' + ',' +'32' + ',' 'Regular'+ ',' + '0.25' '] '
    cmd = cmd + '--convergence [1000x500x250x100,1e-6,10] --shrink-factors 12x8x4x2 --smoothing-sigmas 4x3x2x1vox --transform Affine[0.1] '
    cmd = cmd + '--metric MI'
    cmd = cmd + '[' + fixedImage + ',' + movingImage + ',' + '1' + ',' +'32' + ',' 'Regular'+ ',' + '0.25' '] '
    cmd = cmd + '--convergence [1000x500x250x100,1e-6,10] --shrink-factors 12x8x4x2 --smoothing-sigmas 4x3x2x1vox --transform SyN[0.1,3,0] --metric CC'
    #cmd = cmd + '--convergence [1000x500x250x100,1e-6,10] --shrink-factors 12x8x4x2 --smoothing-sigmas 4x3x2x1vox --transform BSplineSyn[0.1,3,0] --metric CC'
    cmd = cmd + '[' + fixedImage + ',' + movingImage + ',' + '1'+ ',' + '4' '] '
    cmd = cmd + '--convergence [100x100x70x50x20,1e-6,10] --shrink-factors 10x6x4x2x1 --smoothing-sigmas 5x3x2x1x0vox'
    output = subprocess.check_call(cmd, shell = True)

def runantsApplyTransforms(fixedImage, movingImage, affine, warp, output):
    #fixedImage = image being registered to
    #movingImage = image being registered
    #strAffine = matches part of the string the Affine file
    #strWarp = matches a part of the string
    #fixedImage=fixedImage[0]
    #code.interact(local=locals())
    #return fixedImage, movingImage, strAffine, strWarp
    
    cmd = '/groups/flyfuncconn/home/busheyd/antsbin/bin/antsApplyTransforms'
    #targetdir = movingImage[0 : movingImage.rfind('/')+1]
    #find the affine file,
    #affinefile, indx2=osDB.getFileContString(targetdir, strAffine) 
    #affine = targetdir + affinefile.values[0]
    #find warp file
    #warpfile, indx2 = osDB.getFileContString(targetdir, strWarp)
    #warp = targetdir + warpfile.values[0]   
    #transformed = targetdir + 'Transformed.nii.gz'
    cmdL = cmd + ' -d 3 -i ' + movingImage[0] + ' -o ' + output[0] + ' -r ' + fixedImage[0] + ' -t ' + warp[0] + ' -t ' + affine[0]
    print cmdL
    subprocess.check_call(cmdL, shell = True)
    return [cmdL]

def modifyHeadersNifti(targetfile, affine, output):
    #targetfile = path the nifti file
    #affine
    #output = file to write modified nifti file
    nifti1 = nibabel.load(targetfile[0])
    newni = nibabel.Nifti1Image(nifti1.get_data(),affine)
    newni.to_filename(output[0])
    #cannot reset dimensions in header because the affine resets it.
    
    
    
def convertTif_Nifti(targetfile, savefile):
    
    #converts path/.tif file into nifti placing the nii file in the same directory
    imgTif = tifffile.imread(targetfile[0])
    imgTif = np.transpose(imgTif, [2, 1, 0])
    affine1 = np.array([[-1.,  0.,  0., -0.], [ 0., -1.,  0., -0.], [ 0.,  0.,  1.,  0.], [ 0.,  0.,  0.,  1.]])
    nifti1 = nibabel.Nifti1Image(imgTif, affine1)
    nifti1.header['pixdim'][1:4] = [0.4596, 0.4596, 5]
    #nifti1.header['pixdim']
    nibabel.save(nifti1, savefile[0])
    #need to check johns fiji nii files to see if they match
    #nifit file generated from tif in imageJ
    #testfile = "/tier2/flyfuncconn/DanB/r3_ol0020blc10_frup65/20160301_r3_ol0020blc10_frup65_rnai_flyb_00002_regression/Baseline2Reference2/20160301_r3_ol0020blc10_frup65_rnai_flyb_00001_baselineFiji.nii"
    testfile = "/tier2/flyfuncconn/DanB/r3_ol0020blc10_frup65/20160301_r3_ol0020blc10_frup65_flya_00002_regression/RegistrationNSync2/20160301_r3_ol0020blc10_frup65_flya_00002_PyReg1InverseWarp.nii.gz"
    testfile ="/tier2/flyfuncconn/DanB/r3_ol0020blc10_frup65/20160301_r3_ol0020blc10_frup65_flya_00002_regression/RegistrationNSync2/Transformed.nii.gz"
    testfile = "/tier2/flyfuncconn/DanB/r3_ol0020blc10_frup65/20160301_r3_ol0020blc10_frup65_flya_00002_regression/MY20160301_r3_ol0020blc10_frup65_flya_00002_baseline_spacingWarp.nii.gz"
    nifti1 = nibabel.load(testfile)
    print nifti1.header
    print nifti1.get_affine()
    #John's image file
    #testfile = ''
    #t=nibabel.load(testfile)
    
def loadImgArray(path2Img):

    path, filename = os.path.split(path2Img)
    if filename.find('.nii') > 0:
        img1 = nibabel.load(path2Img)
        output1 = img1.get_data()
    elif filename.find('.tif') > 0:
        output1 = tifffile.imread(path2Img)
        if len(output1.shape) == 3:
            output1 = np.transpose(output1, [2, 1, 0])
    elif filename.find('.nrrd') > 0:
        output1, _ = nrrd.read(path2Img)
    else:
        print 'File format not recognized'
    #normalize image
    output1 = output1.astype('float16') / output1.astype('float16').max()
       
    return output1 
def loadImgArray2(path2Img):

    path, filename = os.path.split(path2Img)
    if filename.find('.nii') > 0:
        img1 = nibabel.load(path2Img)
        output1 = img1.get_data()
    elif filename.find('.tif') > 0:
        output1 = tifffile.imread(path2Img)
        if len(output1.shape) == 3:
            output1 = np.transpose(output1, [2, 1, 0])
    elif filename.find('.nrrd') > 0:
        output1, _ = nrrd.read(path2Img)
    else:
        print 'File format not recognized'
    return output1          
     
def ConvertTransformFile(target, output):
    #convert affine.mat files generated by antsRegistrationNSyncQuick to a format that antsBigWarpTransform.java can use
    cmd = '/groups/flyfuncconn/home/busheyd/antsbin/bin/ConvertTransformFile 3 '
    cmdL =cmd + target[0] + ' ' + output[0] +' --hm'       
    print cmdL
    result1 = subprocess.check_call(cmdL, shell = True)     
    return result1
'''   
def reSaveNIFTI(target, output):
target1 = target[0][0]
output1 = output[0][0]


#convert nii.gz files produced by ants to a nii format that antsBigWarpTransform.java can use
tarimg = nibabel.load(target1)
nibabel.save(tarimg, output1)
'''

def runMIPAVsaveas(target, output):
    target1 = target[0]
    output1 = output[0]
    #return [target, output]
    #import pdb; pdb.set_trace()
    cmdL = '/groups/flyfuncconn/home/busheyd/scripts/Python/Modules/mipavResave'
    print target1
    print output1
    return1 = subprocess.check_call([cmdL, target1, output1])
    return return1
    

def genMagnetaCompositeImage(magImg, greenImg, output=False):
	#generates RGB image with magImg having magneta hue and greenImg green hue
	#magImg = file/path to mageneta
	#greenImg = file/path to green image
	#output = file/path to save composite image
	#import pdb; pdb.set_trace()
	from skimage.exposure import adjust_gamma
	img1 = loadImgArray(magImg[0])
	img1 = adjust_gamma(img1, 0.5)
	img2 = loadImgArray(greenImg[0])
	img1 = hist_match(img1, img2)
	RGB = np.zeros( [img1.shape[2], img1.shape[1], img1.shape[0],  3],dtype=np.float)
	RGB[:,:,:, 0] = np.rollaxis(np.rollaxis(img1, 1), 2)
	RGB[:,:,:, 1] = np.rollaxis(np.rollaxis(img2, 1), 2)
	RGB[:,:,:, 2] = np.rollaxis(np.rollaxis(img1, 1), 2)
	RGB = np.array(RGB*255).astype('uint8')
	#shape must equal Z, y, x, rgb
	if output:
		tifffile.imsave(output[0], RGB)
	else:
		return RGB

def geneMagnetaCompoisteImageSideBy(magImg, greenImg, output=False):
	#generates RGB image with magImg having magneta hue and greenImg green hue
	#magImg = file/path to mageneta
	#greenImg = file/path to green image
	#output = file/path to save composite image
	#import pdb; pdb.set_trace()
	from skimage.exposure import adjust_gamma
	img1 = loadImgArray(magImg[0])
	#img1 = adjust_gamma(img1, 0.5)
	img2 = loadImgArray(greenImg[0])
	img1 = hist_match(img1, img2)
	RGB = np.zeros( [img1.shape[2], img1.shape[1], img1.shape[0],  3],dtype=np.float)
	RGB[:,:,:, 0] = np.rollaxis(np.rollaxis(img1, 1), 2)
	RGB[:,:,:, 1] = np.rollaxis(np.rollaxis(img2, 1), 2)
	RGB[:,:,:, 2] = np.rollaxis(np.rollaxis(img1, 1), 2)
	Mag = np.concatenate((generateWhiteRGB(img1), generateWhiteRGB(img2)), axis=2)
	Mag = np.concatenate((Mag, RGB), axis = 2)
	#shape must equal Z, y, x, rgb
	Mag = np.array(Mag*255).astype('uint8')
	if output:
		tifffile.imsave(output[0], Mag)
	else:
		return Mag

def generateWhiteRGB(img1):
	#provide numpy array with black and wite image
	RGB=np.zeros( [img1.shape[2], img1.shape[1], img1.shape[0], 3],dtype=img1.dtype)
	for c in range(RGB.shape[3]):
		RGB[:, :, :, c] = np.rollaxis(np.rollaxis(img1, 1), 2)
	return RGB   
          
def getImageData(imgFile):
    #imgFile should be a list of paths to target image files
    imgData = pd.DataFrame(index = imgFile, columns = ['datatype', 'width', 'height', 'depth', 'max-intensity', 'min-intensity'])
    for file1 in imgData.index:
        img = tifffile.imread(file1)
        imgData['datatype'].loc[file1] = img.dtype
        imgData['width'].loc[file1] = img.shape[1]
        imgData['height'].loc[file1] = img.shape[2]
        imgData['depth'].loc[file1] = img.shape[0]
        imgData['max-intensity'].loc[file1] = np.nanmax(img)
        imgData['min-intensity'].loc[file1] = np.nanmin(img)
    return imgData

def findPilTag(imgPil, tagStr):
    #designed get to metda data from tif
    #examples tagStr
    #time image started = frameTimestamps_sec
    #find channel offset = 'scanimage.SI.hChannels.channelOffset'
    #tagstr = ChannelOffset
    #imgPil = loaded Pil image cimg = Image.open(path + '/' + cfile)
    #tagStr = 'string' to search for in the tags
    for key1 in imgPil.tag.keys():
        ctag= imgPil.tag[key1][0]
        #print ctag
        if isinstance(ctag, unicode):
            for line1 in ctag.split('\n'):
                #print line1
                if tagStr in line1:
                    return line1[len(tagStr)+3:]
                    

class fluorescent():
    #used to study fluorescent data
    # data = shape[animals , times]
    def __init__(self, data, offset, start, response, framerate):
        self.data = data            # data is fluorescent data (trial, time)
        self.offset = offset        #offset should be the median of all offset data
        self.start = start          #start = when stimulation started
        self.response = response    #expected response either 'neg' (reduced fluorescence) or 'pos' (increased response to stimulus
        self.framerate=framerate
        
    def deltaFF(self):
        data1 = self.data - self.offset
        preF = np.mean(data1[:, 0 : self.start], axis=1)
        dF = data1.T - preF
        dFF = dF/preF
        for row in range(dFF.shape[0]):
            dFF[row,:] = nd.filters.gaussian_filter1d(dFF[row,:], 1)
        return dFF.T 
    
    def offsetCorrected(self):
        return self.data - self.offset
        
    def removeSign(self):
        data1 = self.deltaFF()
        if self.response is 'neg':
            data1[data1>0] = np.NaN
        elif self.response is 'pos':
            data1[data1<0] = np.NaN
        else:
            data1 = np.absolute(data1)
        return data1
    
    def Max(self): 
        data1 = self.removeSign()
        return np.nanmax(np.absolute(data1[:, self.start:]), axis = 1)   
    
    def Mean(self):
        data1 = self.removeSign()
        return np.nanmean(np.absolute(data1[:, self.start:]), axis = 1)
    
    def SNR(self): #signal to noise ratio
        data1 = self.removeSign()
        return np.nanmax(np.absolute(data1[:, self.start:]), axis = 1) / np.nanstd(np.absolute(data1[:, :self.start]), axis =1)
    
    def delay2Max(self):
        data1 = self.removeSign()
        t1=np.empty(self.data.shape[0])
        for i in range(self.data.shape[0]):
            if not np.all(np.isnan(data1[i, self.start:])):
                t1[i] = np.nanargmax(np.absolute(data1[i, self.start:]))
            else:
                t1[i] = np.nan
        return t1*self.framerate
    
    def Median(self):
        data1 = self.removeSign()
        return np.nanmedian(np.absolute(data1[:, self.start:]), axis = 1)
    
def saveNifti(data, pathfile):
    if isinstance(data, str):
        data = tifffile.imread(data)
    data = np.transpose(data, [2, 1, 0])
    affine1 = np.array([[-1.,  0.,  0., -0.], [ 0., -1.,  0., -0.], [ 0.,  0.,  1.,  0.], [ 0.,  0.,  0.,  1.]])
    nifti1 = nibabel.Nifti1Image(data, affine1)
    nibabel.save(nifti1, pathfile)

def saveNiftiFromNifti(data, pathfile):
    affine1 = np.array([[-1.,  0.,  0., -0.], [ 0., -1.,  0., -0.], [ 0.,  0.,  1.,  0.], [ 0.,  0.,  0.,  1.]])
    nifti1 = nibabel.Nifti1Image(data, affine1)
    nibabel.save(nifti1, pathfile)

def generateAntsTemplate(TemplateGeneration): 
    #TemplateGeneration = pd.DataFrame(index = ['ShortBaseline', 'Baseline'], columns = ['OutputPath', 'InputFiles', 'TimeStart', 'TimeEnd', 'CommandLine', 'TransformationMethod'] )
    #index = name to place in dataframe
    #InputFiles = ['path/img1' ....]
    #OutputPath = path to output files /'T_'  
    #TransformationMethod = 'Syn' or '  BSplineSyN[0.1,75,0]'
    #create a string containing target files
    for cindex in TemplateGeneration.index:
        inputfiles = '' #need to generate string of files to generate template in antsMultivariateTemplateConsctruion2.sh
        for row in TemplateGeneration['InputFiles'].loc[cindex]:
            inputfiles = inputfiles + ' ' + row
        
        if not os.path.isdir(TemplateGeneration['-o'].loc[cindex][:-3]):
            os.mkdir(TemplateGeneration['-o'].loc[cindex][:-3])
        TemplateGeneration['CommandLine'].loc[cindex] = "${ANTSPATH}/antsMultivariateTemplateConstruction2.sh -d 3 -o " + TemplateGeneration['-o'].loc[cindex] +  " -i " + TemplateGeneration['-i'].loc[cindex] + " -g " + TemplateGeneration['-g'].loc[cindex] + ' -j ' + TemplateGeneration['-j'].loc[cindex] + " -c " + TemplateGeneration['-c'].loc[cindex] + " -f " + TemplateGeneration['-f'].loc[cindex] + " -s " + TemplateGeneration['-s'].loc[cindex] + " -q " + TemplateGeneration['-q'].loc[cindex] + " -n " + TemplateGeneration['-n'].loc[cindex] + " -r " + TemplateGeneration['-r'].loc[cindex] + " -m " + TemplateGeneration['-m'].loc[cindex] + " -t " + TemplateGeneration['-t'].loc[cindex]  + " " + inputfiles[1:]
        TemplateGeneration['TimeStart'].loc[cindex] = time.time()
        output = subprocess.check_call(TemplateGeneration['CommandLine'].loc[cindex], shell = True)
        TemplateGeneration['TimeEnd'].loc[cindex] = time.time()
    return TemplateGeneration
    '''
    TemplateGeneration['-t'] = 'SyN'
    TemplateGeneration['-i'] = '4'
    TemplateGeneration['-g'] = '0.15'
    TemplateGeneration['-j'] = '8'
    TemplateGeneration['-c'] = '0' 
    TemplateGeneration['-k'] = '1' 
    TemplateGeneration['-w'] = '1'
    TemplateGeneration['-f'] = '6x4x2'
    TemplateGeneration['-s'] = '2x1x0'
    TemplateGeneration['-q'] = '25x45x10'
    TemplateGeneration['-n'] = '0'
    TemplateGeneration['-r'] = '0'
    TemplateGeneration['-m'] = 'CC'
    '''
def generateAntsTemplateGenCMD(TemplateGeneration): 
    #TemplateGeneration = pd.DataFrame(index = ['ShortBaseline', 'Baseline'], columns = ['OutputPath', 'InputFiles', 'TimeStart', 'TimeEnd', 'CommandLine', 'TransformationMethod'] )
    #index = name to place in dataframe
    #InputFiles = ['path/img1' ....]
    #OutputPath = path to output files /'T_'  
    #TransformationMethod = 'Syn' or '  BSplineSyN[0.1,75,0]'
    #create a string containing target files
    for cindex in TemplateGeneration.index:
        inputfiles = '' #need to generate string of files to generate template in antsMultivariateTemplateConsctruion2.sh
        for row in TemplateGeneration['InputFiles'].loc[cindex]:
            inputfiles = inputfiles + ' ' + row
        
        if not os.path.isdir(TemplateGeneration['-o'].loc[cindex][:-3]):
            os.mkdir(TemplateGeneration['-o'].loc[cindex][:-3])
        TemplateGeneration['CommandLine'].loc[cindex] = "${ANTSPATH}/antsMultivariateTemplateConstruction2.sh -d 3 -o " + TemplateGeneration['-o'].loc[cindex] +  " -i " + TemplateGeneration['-i'].loc[cindex] + " -g " + TemplateGeneration['-g'].loc[cindex] + ' -j ' + TemplateGeneration['-j'].loc[cindex] + " -c " + TemplateGeneration['-c'].loc[cindex] + " -f " + TemplateGeneration['-f'].loc[cindex] + " -s " + TemplateGeneration['-s'].loc[cindex] + " -q " + TemplateGeneration['-q'].loc[cindex] + " -n " + TemplateGeneration['-n'].loc[cindex] + " -r " + TemplateGeneration['-r'].loc[cindex] + " -m " + TemplateGeneration['-m'].loc[cindex] + " -t " + TemplateGeneration['-t'].loc[cindex]  + " " + inputfiles[1:]
        TemplateGeneration['TimeStart'].loc[cindex] = time.time()
        #output = subprocess.check_call(TemplateGeneration['CommandLine'].loc[cindex], shell = True)
        TemplateGeneration['TimeEnd'].loc[cindex] = time.time()
    return TemplateGeneration
    '''
    TemplateGeneration['-t'] = 'SyN'
    TemplateGeneration['-i'] = '4'
    TemplateGeneration['-g'] = '0.15'
    TemplateGeneration['-j'] = '8'
    TemplateGeneration['-c'] = '0' 
    TemplateGeneration['-k'] = '1' 
    TemplateGeneration['-w'] = '1'
    TemplateGeneration['-f'] = '6x4x2'
    TemplateGeneration['-s'] = '2x1x0'
    TemplateGeneration['-q'] = '25x45x10'
    TemplateGeneration['-n'] = '0'
    TemplateGeneration['-r'] = '0'
    TemplateGeneration['-m'] = 'CC'
    '''

def convertValues2List(list1):
    return [[path] for path in list1]

def loadNifti(pathfile):
    img = nibabel.load(pathfile)
    return img.get_data()

def runBashCommand(bashcommand):
    out1 = [time.time()]
    #bashcommand = 'echo Hellow World'
    out1.append( subprocess.check_call(bashcommand, shell = True))
    out1.append(time.time())
    return out1

def RegisterImages(path, outputpathfile, slice1=42, limit=30):
    tiffiles, indx = osDB.getFileContString(path, 'tif')
    #load first 30 tifffiles
    img1 = tifffile.imread(os.path.join(path, tiffiles.sort_values().iloc[0]))
    img2 = np.zeros(([30, img1.shape[0]-3, img1.shape[1], img1.shape[2]]), dtype = img1.dtype)
    img2[0, :, :, :] = img1[:slice1, :, :]
    #img10 = td.images.fromtif(os.path.join(outputdata['Raw_Folder'].loc[cindex], '')
    for i, imgfile in enumerate(tiffiles.sort_values().iloc[1:limit]):
        #print(os.path.join(outputdata['Raw_Folder'].loc[cindex], imgfile))
        img2[i+1, : ,: ,:] = tifffile.imread(os.path.join(path, imgfile))[:slice1, :, :]
    
    reg = registration.CrossCorr()
    reference = img2.mean(axis=0)
    reference = scipy.signal.medfilt(reference)
    registrationModel = reg.fit(img2, reference = reference)
    #displacements = registrationModel.toarray()
    images = registrationModel.transform(img2)
    #images2 = images.toarray()
    meanImg = images.mean()
    #images.cache()
    saveNifti(meanImg, outputpathfile)

def filteredMask(inputimage, outputimage, threshold = 360, radius1 = 3, radius2 = 7):
    img = loadNifti(inputimage)
    #threshold = skfilters.threshold_otsu(img2)
    img2 = np.zeros(img.shape)
    img2[img >= threshold] = 1
    #opening to remove outside elements
    img3 = np.zeros(img2.shape, dtype = img2.dtype)
    for layer in range(img2.shape[2]):
        img3[:, :, layer] = skmorphology.binary_closing(img2[:, :, layer], skmorphology.disk(radius1))
    #tifffile.imsave(outputdata['Mask'].loc[cindex], img3.astype('uint8'))    
    img4 = np.zeros(img2.shape, dtype = img2.dtype)
    for layer in range(img2.shape[2]):
        img4[:, :, layer] = skmorphology.binary_opening(img3[:, :, layer], skmorphology.disk(radius2))
    img[~img4.astype('bool')] = 0
    saveNiftiFromNifti(img, outputimage)
    
    #img2 = skmorphology.closing
def removeSliceNifti(inputfile, outputfile, slice1):
    img = loadNifti(inputfile)
    img = img[:, :, :slice1]
    saveNiftiFromNifti(img, outputfile)
    
def convertNifti2Tiff(inputfile):
    img = loadNifti(inputfile)
    img = np.transpose(img, [2, 1, 0])
    img= img.astype('uint16')
    tifffile.imsave(inputfile[:-7] +'.tif', img)
    
def genJacImage(inputWarp, outputJac):
    #generates an image highlighting local deformation
    antsCMD = '${ANTSPATH}/ANTSJacobian 3 ' + inputWarp + ' ' + outputJac[:-7]
    runBashCommand(antsCMD)
    
def saveTimeSeries(image, filename, outputdir):
    #save each time as separate stacked tif
    #image =numpy array
    #filename for each file
    # outputdir - storage location
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
    for ctime in range(image.shape[0]):
        tifffile.imsave(os.path.join(outputdir, filename + '-' + '{:05d}'.format(ctime)) + '.tif', image[ctime, :, :, :] )
    
    
  
