import skimage
import nibabel
import nrrd
import scipy
import numpy as np
import tifffile
import skimage.morphology


    
def intersectROI(path_roi1, path_roi2, outputfile):
    #path_roi1 and path_roi2 are lists containing paths to roi files 
    #listRoisPaths = path_roi1
    roi1 = combineROI(path_roi1)
    
    roi2 = combineROI(path_roi2)
    combined = roi1 + roi2
    intersectvalue = len(np.where(combined.flatten() > 1 )[0]) / len(np.where(roi1.flatten()> 0)[0])
    combined[combined < 2] = 0
    combined[combined > 0 ] = 1
    #nrrd.write(outputfile, combined)
    tifffile.imsave(outputfile, skimage.img_as_ubyte(combined))
    return intersectvalue

def combineROI(listRoisPaths):
    if '.gz' in listRoisPaths[0]:
        img1 = nibabel.load(listRoisPaths[0]).get_data()
        for cfile in listRoisPaths[1:]:
            img1 = img1 + nibabel.load(cfile).get_data()
        #nrrd file loads dimensionality differntly
        img1 = np.rollaxis(img1, 0, 3)
        img1 = np.rollaxis(img1, 1, 0)
        #original data is not binary
        img1[img1 >0 ] =1
        #fill in holes
        img1 = maskfill(img1)
    else:
        img1 = tifffile.imread(listRoisPaths[0])
        for cfile in listRoisPaths[1:]:
            img1 = img1 + tifffile.imread(cfile)
        #for an unknown reason sometimes tifffile is loading as memmap and need to convert to regular array
        if isinstance(img1, np.core.memmap):
            img2 = np.zeros(img1.shape, dtype=img1.dtype)
            for int in range(img2.shape[0]):
                print(int)
                img2[int, :, :] = img1[int, :, :]
            img1 = img2
        img1[img1>0] = 1
        #img1 = maskfill(img1)
    return img1

def maskfill(img1):
    #file = path to mask file (.tif)
    filter1 = skimage.morphology.disk(15)
    for slice in range(img1.shape[0]):
        img1[slice, :, :] = skimage.morphology.binary_closing(img1[slice, :, :], filter1)
    img1 = scipy.ndimage.morphology.binary_closing(img1)
    return img1
