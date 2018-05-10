# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 23:33:16 2016

@author: Surf32
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib import path
from matplotlib import lines
from  skimage import exposure
from skimage.measure import regionprops
from skimage.measure import label
import skimage
from numpy.lib import arrayterator
import os
import tifffile
import sys
from PyQt4 import QtGui
#from PyQt5 import QtGui, QtWidgets
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import Lasso
import matplotlib
import skimage.measure
import os
from  skimage import exposure


def startRoiSelectionTool(filelist, rois):
    app = QtGui.QApplication(sys.argv)
    main = roiSelectionTool(filelist, rois)
    main.show()
    sys.exit(app.exec_())

#class roiSelectionTool(QtWidgets):
class roiSelectionTool(QtGui.QWidget):
    def __init__(self, filelist, rois, parent=None):
        super(roiSelectionTool, self).__init__(parent)
        self.rois = rois
        # a figure instance to plot on
        self.figure = plt.figure()
        self.filelist = filelist
        self.currentImage = 0 # number in list of current images
        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        
        #generate list widget
        self.listW = QtGui.QListWidget()
        self.listW.addItems(self.rois)
        self.colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (1, 0.55, 0.55), (0.55,0.55,0.55), (0.5, 0, 0), (1, 0.5, 0)]
        for index in xrange(self.listW.count()): 
            self.listW.item(index).setTextColor(QtGui.QColor(self.colors[index][0]*255, self.colors[index][1]*255, self.colors[index][2]*255))
        self.listW.setCurrentRow(0)
        #self.setMaximumWidth(10000)
        
        #generate canvas
        self.canvas = FigureCanvas(self.figure)
        self.cid1 = self.canvas.mpl_connect('button_press_event', self.onPressCanvas)
        #self.canvas.setMaximumSize(1000000, 10000000)
        
        #generate scrollbar
        self.scroll=QtGui.QScrollBar()
        self.scroll.valueChanged.connect(self.changeSlice)
        self.scroll.setMaximum(255)
        self.scroll.setOrientation(1)
        self.scroll.setMaximumSize(300000, 50)
        
        #self.scroll.sliderMoved.connect(1)
        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        #radio buttons choosing image type
        self.imageTypeBox = QtGui.QGroupBox('Choose Image')
        imagetypes = ['StDev', 'Max', 'Sum', 'Stack']
        radioboxForm = QtGui.QFormLayout()
        self.radioImageType = {}
        for ctype in imagetypes:
            cr = QtGui.QRadioButton(ctype)
            cr.clicked.connect(self.updateImage)
            self.radioImageType[ctype] = cr
            radioboxForm.addRow(cr)
            if ctype == 'StDev':
                cr.setChecked(1)
        self.imageTypeBox.setLayout(radioboxForm)
        #import pdb; pdb.set_trace()
        #self.imageTypeBox.clicked.connect(self.currentImageType)
        #self.imageTypeBox.mousePressEvent(self.currentImageType)
        
        #radio buttons choosing iROI tool
        self.roiTypeBox = QtGui.QGroupBox('Choose ROI')
        roiTypes = ['lasso', 'polygon', 'circle', 'square']
        radioboxForm = QtGui.QFormLayout()
        self.radioRoiTypes = {}
        for ctype in roiTypes:
            cr = QtGui.QRadioButton(ctype)
            self.radioRoiTypes[ctype] = cr
            radioboxForm.addRow(cr)
            if ctype == 'lasso':
                cr.setChecked(1)
        self.roiTypeBox.setLayout(radioboxForm)
        

        # buttons to change image currently viewed
        self.nextImgB = QtGui.QPushButton('Next Image')
        self.nextImgB.clicked.connect(self.nextImage)
        self.nextImgB.setMaximumSize(200,50)
        self.prevImgB = QtGui.QPushButton('Previous Image')
        self.prevImgB.clicked.connect(self.prevImage)
        self.prevImgB.setMaximumSize(200,50)
        self.eraseMaskB = QtGui.QPushButton('Erase Mask')
        self.eraseMaskB.clicked.connect(self.eraseMask)
        self.eraseMaskB.setMaximumSize(200,50)
        
        # buttons to control contrast
        self.contrastB = QtGui.QPushButton('Contrast')
        self.contrastB.clicked.connect(self.contrast)
        self.contrastB.setMaximumSize(200,50)
        
        # set the layout
        self.grid = QtGui.QGridLayout()
        self.grid.setSpacing(5)
        #first is row and next is column
        self.grid.addWidget(self.toolbar, 0,0)
        self.grid.addWidget(self.canvas, 1,0, 4,3)
        self.grid.addWidget(self.nextImgB, 0,2)
        self.grid.addWidget(self.prevImgB, 0,1)
        self.grid.addWidget(self.scroll, 5,0, 5,3)
        self.grid.addWidget(self.listW, 0,3, 2, 3)
        self.grid.addWidget(self.imageTypeBox, 3,3)
        self.grid.addWidget(self.roiTypeBox, 4,3)
        self.grid.addWidget(self.eraseMaskB, 4,4)
        self.grid.addWidget(self.contrastB, 5,4)
        self.grid.setColumnStretch(1, 4)
        self.grid.setRowStretch(3, 1)
        #self.grid.setColumnStretch()
        self.setLayout(self.grid)
        
        self.showMaximized()
        self.LoadImage()
        
    def eraseMask(self):
        savefile = os.path.split(self.imgpath)
        savefile = os.path.join(savefile[0], str(self.listW.currentItem().text()) + '_Mask.npy')
        print('Deleting_' + savefile)
        os.remove(savefile)
        self.drawROI()
        
        
    def saveMask(self):
        savefile = os.path.split(self.imgpath)
        savefile = os.path.join(savefile[0], str(self.listW.currentItem().text()) + '_Mask.npy')
        print('Saving_' + savefile)
        np.save(savefile, self.IndexMask)
       
    
    def onPressCanvas(self, event):
        print("Event detected starting whatever tool is selected")
        self.canvas.mpl_disconnect(self.cid1)
        if self.canvas.widgetlock.locked():
            return
        if event.inaxes is None:
            return
        if self.radioRoiTypes['lasso'].isChecked():
            self.lasso = LassoDB(event.inaxes, (event.xdata, event.ydata), self.generateMask)
        elif self.radioRoiTypes['polygon'].isChecked():
            #self.polygon = polygonCreator2(self.canvas, self.ax, (event.xdata, event.ydata), self.generateMask)
            self.polygon = polygonCreator(event.inaxes,  (event.xdata, event.ydata), self.generateMask)
        else:
            print('Tool not implemented')
        
        # acquire a lock on the widget drawing
        #self.canvas.widgetlock(self.lasso)

        
    def generateMask(self, verts):
        p = matplotlib.lines.Path(verts)
        pix1 = np.arange(self.img.shape[1])
        pix2 = np.arange(self.img.shape[2])
        xv, yv = np.meshgrid(pix1,pix2)
        pix = np.vstack( (xv.flatten(), yv.flatten()) ).T
        ind = p.contains_points(pix, radius=1)
        savefile = os.path.split(self.imgpath)
        savefile = os.path.join(savefile[0], str(self.listW.currentItem().text()) + '_Mask.npy')
        if os.path.isfile(savefile):
            index = np.load(savefile)
            self.Mask = np.zeros([self.img.shape[1], self.img.shape[2]])
            if len(index) != 0:
                self.Mask[(index[0], index[1])] = 1 
        else:
            self.Mask = np.zeros([self.img.shape[1], self.img.shape[2]])
        lin = np.arange(self.Mask.size)
        newArray = self.Mask.flatten()
        newArray[lin[ind]] = 1
        #newArray[indices] = 1
        self.Mask = newArray.reshape(self.Mask.shape)
        self.IndexMask = np.where(self.Mask == 1)
        self.saveMask()
        self.drawROI()
        self.cid1 = self.canvas.mpl_connect('button_press_event', self.onPressCanvas)
        #del self.lasso
 
        
    def drawROI(self):
        #remove old plot
        if 'plottedlines' in dir(self):
            for i, line1 in enumerate(self.plottedlines):
                #import pdb; pdb.set_trace()
                self.ax.lines.remove(line1[0])
                #del self.ax.lines(i)
                #line1.remove()
        
        #plot masks
        self.plottedlines = []
        for cn, cmask in enumerate(self.rois):
            savefile = os.path.split(self.imgpath)
            savefile = os.path.join(savefile[0], cmask + '_Mask.npy')
            #import pdb; pdb.set_trace()
            if os.path.isfile(savefile):
                index = np.load(savefile)
                self.Mask = np.zeros([self.img.shape[1], self.img.shape[2]])
                if len(index)!=0:
                    self.Mask[(index[0], index[1])] = 1
                    contours = skimage.measure.find_contours(self.Mask, 0.8)
                    for n, contour in enumerate(contours):
                        self.plottedlines.append(self.ax.plot(contour[:, 1], contour[:, 0], linewidth = 2, color = self.colors[cn]))
                    #msk.set_data(array)
                    self.ax.axis('on')
                    self.ax.set(adjustable = "datalim")
                    self.ax.set_ylim([self.img.shape[2], 0])
                    self.ax.set_xlim([0, self.img.shape[1]])
                    #self.imagehandle.ylim([0, 512])
                    #self.ax.xlim([0, 512])
                    #self.ax.set(adjustable = 'datalim')
        self.figure.canvas.draw_idle()
                
 
            

    def LoadImage(self):

        self.imgpath = self.filelist[self.currentImage]
        #self.cimg +=1
        print('loading Image')
        print(self.imgpath)
        self.img = tifffile.imread(self.imgpath)
        print('Finished Loading Image')
        #self.img=exposure.adjust_gamma(self.img, 0.6)
        self.scroll.setMaximum(self.img.shape[0])  #set maximum number of images for scroll bar
        # create an axis
        self.ax = self.figure.add_axes([0,0,1,1])
        self.ax.axis('off')
        
        
        # discards the old graph
        #self.ax.hold(False)

        # plot data
        self.currentSlice = 0
        #self.imagehandle = self.ax.imshow(np.std(self.img, axis=0), extent = [0, self.img.shape[1], 0, self.img.shape[2]]) #use extent to try to prevent resizing after plotting mask but only prevents resizing in Y
        #self.imagehandle = self.ax.imshow(np.std(self.img, axis=0))
        self.imagehandle = self.ax.imshow(np.std(self.img, axis=0), aspect = 'equal', cmap='Greys_r') 
        self.ax.set(adjustable = "datalim")
        print('Xaxis_ ' + str(self.ax.get_xlim()))
        print('Yaxis_ ' + str(self.ax.get_ylim()))
        if 'text1' in dir(self):
            self.text1.set_text(self.imgpath[-50:])
        else:
            self.text1 = self.ax.text(self.ax.get_xlim()[0]+10, self.ax.get_ylim()[1]+10, self.imgpath[-50:], color = (1,1,1), verticalalignment = 'bottom', horizontalalignment ='left')
        #self.ax.text(250, 250, self.imgpath[-15], color = (1,1,1), verticalalignment = 'bottom', horizontalalignment ='right')
        
        #draw in masks
        self.drawROI()
        self.canvas.draw()
        
    def updateImage(self):
        print('updatingimage')
        #import pdb; pdb.set_trace()
        if self.radioImageType['Stack'].isChecked():
            self.imagehandle.set_data(self.img[self.currentSlice, :, :])
            self.imagehandle.set_clim(np.min(self.img), np.max(self.img))
        elif self.radioImageType['StDev'].isChecked():
            self.imagehandle.set_data(np.std(self.img, axis=0))
            self.imagehandle.set_clim(np.min(np.std(self.img, axis=0)), np.max(np.std(self.img, axis=0)))
        elif self.radioImageType['Max'].isChecked():
            self.imagehandle.set_data(np.max(self.img, axis=0))
            self.imagehandle.set_clim(np.min(self.img), np.max(self.img))
        elif self.radioImageType['Sum'].isChecked():
            self.imagehandle.set_data(np.sum(self.img, axis=0))
            self.imagehandle.set_clim(np.min(np.sum(self.img, axis=0)), np.max(np.sum(self.img, axis=0)))
        self.canvas.draw()

        
        
    def changeSlice(self):
        self.currentSlice = self.scroll.value()
        self.updateImage()
        
        
    def prevImage(self):
        self.currentImage-=1
        if self.currentImage >= len(self.filelist):
            self.currentImage = len(self.filelist)-1
        self.LoadImage()
    
    def nextImage(self):
        self.currentImage+=1
        if self.currentImage<0:
            self.currentImage = 0
        self.LoadImage()
        
    def contrast(self):
        print('Not finished')
        self.cid1 = self.canvas.mpl_connect('button_press_event', self.onPressCanvas)
        #plan to generate a separate gui to control contrast
        
        
class polygonCreator(matplotlib.widgets.AxesWidget):
    #stylized off the lasso class in widgets matplotlib
    def __init__(self, ax, xy, callback=None, useblit=True):
        matplotlib.widgets.AxesWidget.__init__(self, ax)
        
        self.useblit = useblit and self.canvas.supports_blit
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        x, y = xy
        self.verts = [(x, y)]
        self.line = plt.Line2D([x], [y], linestyle='-', color='b', lw=2)
        self.ax.add_line(self.line)
        self.callback = callback
        self.connect_event('button_press_event', self.storeCoord)
        #self.connect_event('motion_notify_event', self.onmove)
        #self.connect_event('button_press_event', self.onmove)    
        
    def storeCoord(self, event):
        
        if event.dblclick:
            print('double click')
            self.verts.append((event.xdata, event.ydata))
            self.callback(self.verts)
            self.ax.lines.remove(self.line)
            self.disconnect_events()
        else:
            print('click', event)
            if event.inaxes != self.ax.axes: return
            #if this the first point
            self.verts.append((event.xdata, event.ydata))
            self.line.set_data(list(zip(*self.verts)))
            #self.line.figure.canvas.draw()
            if self.useblit:
                self.canvas.restore_region(self.background)
                self.ax.draw_artist(self.line)
                self.canvas.blit(self.ax.bbox)
            else:
                self.canvas.draw_idle()
                
    #def onmove(self, event):
        #print('x-' + str(event.xdata))

        
class LassoDB(matplotlib.widgets.AxesWidget):
    """Selection curve of an arbitrary shape.

    The selected path can be used in conjunction with
    :func:`~matplotlib.path.Path.contains_point` to select data points
    from an image.

    Unlike :class:`LassoSelector`, this must be initialized with a starting
    point `xy`, and the `Lasso` events are destroyed upon release.

    Parameters:

    *ax* : :class:`~matplotlib.axes.Axes`
        The parent axes for the widget.
    *xy* : array
        Coordinates of the start of the lasso.
    *callback* : function
        Whenever the lasso is released, the `callback` function is called and
        passed the vertices of the selected path.

    """

    def __init__(self, ax, xy, callback=None, useblit=True):
        matplotlib.widgets.AxesWidget.__init__(self, ax)

        self.useblit = useblit and self.canvas.supports_blit
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        x, y = xy
        self.verts = [(x, y)]
        self.line = plt.Line2D([x], [y], linestyle='-', color=(0.75, 0.75, 0.75), lw=2)
        self.ax.add_line(self.line)
        self.callback = callback
        self.connect_event('button_release_event', self.onrelease)
        self.connect_event('motion_notify_event', self.onmove)

    def onrelease(self, event):
        if self.ignore(event):
            return
        if self.verts is not None:
            self.verts.append((event.xdata, event.ydata))
            if len(self.verts) > 2:
                self.callback(self.verts)
            self.ax.lines.remove(self.line)
        self.verts = None
        self.disconnect_events()

    def onmove(self, event):
        if self.ignore(event):
            return
        if self.verts is None:
            return
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        self.verts.append((event.xdata, event.ydata))

        self.line.set_data(list(zip(*self.verts)))

        if self.useblit:
            self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.line)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()



def getTimeSeriesData(imgFile, indexFile):
    if isinstance(imgFile, basestring):
        img = tifffile.imread(imgFile)
        
    if isinstance(indexFile, basestring):
        index = np.load(indexFile)
        
    #mask = np.zeros(img.shape)
    #mask[(index[0], index[1]) ] =1
    timeseriesMean = np.empty(img.shape[0])
    for slice1 in range(img.shape[0]):
        csliceI = img[slice1, :, :]
        timeseriesMean[slice1] = np.mean(csliceI[(index[0], index[1]) ])
    return timeseriesMean


def getTimeSeriesDataMultipleROI(imgFile, pathstomask):
    #img = path to img file or img array
    #pathstomask = list of paths to rois 
    img = tifffile.imread(imgFile)
    #mask = np.zeros(img.shape)
    #mask[(index[0], index[1]) ] =1
    timeSeriesMeanList=[]
    for path1 in pathstomask:
        if os.path.isfile(path1):
            index = np.load(path1) 
            timeseriesMean = np.empty(img.shape[0])
            for slice1 in range(img.shape[0]):
                csliceI = img[slice1, :, :]
                timeseriesMean[slice1] = np.mean(csliceI[(index[0], index[1]) ])
            timeSeriesMeanList.append(timeseriesMean)
        else:
            timeseriesMean = np.empty(img.shape[0])
            timeseriesMean[:] = np.nan
            timeSeriesMeanList.append(timeseriesMean)
    return timeSeriesMeanList

def getIndividualPixelsStackedTimeSeriesDataMultipleROI(pathtofiles, pathstomask):
    #img = path to img file or img array
    #pathstomask = list of paths to rois 
    import thunder as td
    rawdata = td.images.fromtif(pathtofiles)
    rawdata = rawdata.toarray()
    slice = 13
    #mask = np.zeros(img.shape)
    #mask[(index[0], index[1]) ] =1
    timeSeriesList=[]
    for path1 in pathstomask:
        if os.path.isfile(path1):
            index = np.load(path1) 
            timeseries = np.empty([len(index[0]), rawdata.shape[0]])
            for cindex in range(len(index[0])):
                try:
                    timeseries[cindex, :] = rawdata[:,slice, index[0][cindex], index[1][cindex]]
                except:
                    import pdb
                    pdb.set_trace()
            timeSeriesList.append(timeseries)
        else:
            timeSeriesList = np.empty(rawdata.shape[0])
            timeSeriesList[:] = np.nan
            timeSeriesList.append(timeseriesMean)
    return timeSeriesList

def viewMask(imgFile, indexFile):
    if isinstance(imgFile, basestring):
        img = tifffile.imread(imgFile)
        
    if isinstance(indexFile, basestring):
        index = np.load(indexFile)
        
    mask = np.zeros([img.shape[1], img.shape[2]])
    mask[(index[0], index[1]) ] =1
    plt.imshow(mask)
    

    
def getTimeSeriesDataMultipleROI2(imgFile, pathstomask):
    #img = path to img file or img array
    #pathstomask = list of paths to rois 
    img = tifffile.imread(imgFile)
    #mask = np.zeros(img.shape)
    #mask[(index[0], index[1]) ] =1
    timeSeriesMeanList=[]
    for path1 in pathstomask:
        if os.path.isfile(path1):
            index = np.load(path1) 
            timeseriesMean = np.empty(img.shape[0])
            for slice1 in range(img.shape[0]):
                csliceI = img[slice1, :, :]
                timeseriesMean[slice1] = np.mean(csliceI[(index[0], index[1]) ])
            timeSeriesMeanList.append(timeseriesMean)
        else:
            timeseriesMean = np.empty(img.shape[0])
            timeseriesMean[:] = np.nan
            timeSeriesMeanList.append(timeseriesMean)
    return timeSeriesMeanList