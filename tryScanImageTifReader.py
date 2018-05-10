'''
Created on May 14, 2017

@author: Surf32
'''
from ScanImageTiffReader import ScanImageTiffReader
import os
file = "C:\\Users\\Surf32\Desktop\\ResearchDSKTOP\\DataJ\\A\\A30_FastROI\\SampleDataComplete\\20161117_r3_nsyb_LC4_RNAi_flyd_L_00001\\20161117_r3_nsyb_LC4_RNAi_flyd_L_00001_00001.tif"
vol=ScanImageTiffReader(file).data();

dir1 = 'C:\\Users\\Surf32\Desktop\\ResearchDSKTOP\\DataJ\\A\\A30_FastROI\\SampleDataComplete'
file1 = '20161117_r3_nsyb_LC4_RNAi_flyd_L_00001_00001.tif'
newpath = os.path.join(dir1, file1)
vol=ScanImageTiffReader(file1).data();