'''
#modified from imageView
Created on Mar 26, 2017

@author: Surf32
'''
# -*- coding: utf-8 -*-
from blaze import nan
"""
ImageView.py -  Widget for basic image dispay and analysis
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more infomation.

Widget used for displaying 2D or 3D data. Features:
  - float or int (including 16-bit int) image display via ImageItem
  - zoom/pan via GraphicsView
  - black/white level controls
  - time slider for 3D data sets
  - ROI plotting
  - Image normalization through a variety of methods
"""
import os
import numpy as np
import pdb
#pdb.set_trace()
import pyqtgraph as pg

from pyqtgraph.Qt import QtCore, QtGui, USE_PYSIDE
if USE_PYSIDE:
    print('pyside')
    from pyqtgraph.imageview.ImageViewTemplate_pyside import *
else:
    #from pyqtgraph.imageview.ImageViewTemplate_pyqtDB import *
    from ImageViewTemplate_pyqtDB import *
    
#from pyqtgraph.graphicsItems.ImageItem import *
from pyqtgraph_ImageItemDB import *
from pyqtgraph.graphicsItems.ROI import *
from pyqtgraph.graphicsItems.LinearRegionItem import *
from pyqtgraph.graphicsItems.InfiniteLine import *
from pyqtgraph.graphicsItems.ViewBox import *
from pyqtgraph.graphicsItems.GradientEditorItem import addGradientListToDocstring
from pyqtgraph import ptime as ptime
from pyqtgraph import debug as debug
from pyqtgraph.SignalProxy import SignalProxy
from pyqtgraph import getConfigOption

try:
    from bottleneck import nanmin, nanmax
except ImportError:
    from numpy import nanmin, nanmax


class PlotROI(ROI):
    def __init__(self, size):
        ROI.__init__(self, pos=[0,0], size=size) #, scaleSnap=True, translateSnap=True)
        self.addScaleHandle([1, 1], [0, 0])
        self.addRotateHandle([0, 0], [0.5, 0.5])

class TimeSeriesView(QtGui.QWidget):
    """
    Widget used for display and analysis of image data.
    Implements many features:
    
    * Displays 2D and 3D image data. For 3D data, a z-axis
      slider is displayed allowing the user to select which frame is displayed.
    * Displays histogram of image data with movable region defining the dark/light levels
    * Editable gradient provides a color lookup table 
    * Frame slider may also be moved using left/right arrow keys as well as pgup, pgdn, home, and end.
    * Basic analysis features including:
    
        * ROI and embedded plot for measuring image values across frames
        * Image normalization / background subtraction 
    
    Basic Usage::
    
        imv = pg.ImageView()
        imv.show()
        imv.setImage(data)
        
    **Keyboard interaction**
    
    * left/right arrows step forward/backward 1 frame when pressed,
      seek at 20fps when held.
    * up/down arrows seek at 100fps
    * pgup/pgdn seek at 1000fps
    * home/end seek immediately to the first/last frame
    * space begins playing frames. If time values (in seconds) are given 
      for each frame, then playback is in realtime.
    """
    sigTimeChanged = QtCore.Signal(object, object)
    sigProcessingChanged = QtCore.Signal(object)
    
    def __init__(self, parent=None, name="ImageView", view=None, imageItem=None, *args):
        """
        By default, this class creates an :class:`ImageItem <pyqtgraph.ImageItem>` to display image data
        and a :class:`ViewBox <pyqtgraph.ViewBox>` to contain the ImageItem. 
        
        ============= =========================================================
        **Arguments** 
        parent        (QWidget) Specifies the parent widget to which
                      this ImageView will belong. If None, then the ImageView
                      is created with no parent.
        name          (str) The name used to register both the internal ViewBox
                      and the PlotItem used to display ROI data. See the *name*
                      argument to :func:`ViewBox.__init__() 
                      <pyqtgraph.ViewBox.__init__>`.
        view          (ViewBox or PlotItem) If specified, this will be used
                      as the display area that contains the displayed image. 
                      Any :class:`ViewBox <pyqtgraph.ViewBox>`, 
                      :class:`PlotItem <pyqtgraph.PlotItem>`, or other 
                      compatible object is acceptable.
        imageItem     (ImageItem) If specified, this object will be used to
                      display the image. Must be an instance of ImageItem
                      or other compatible object.
        ============= =========================================================
        
        Note: to display axis ticks inside the ImageView, instantiate it 
        with a PlotItem instance as its view::
                
            pg.ImageView(view=pg.PlotItem())
        """
        QtGui.QWidget.__init__(self, parent, *args)
        self.markerRadius = 50 #circle radius marking single points/rois
        self.clicked = []
        self.image_Designation = None
        self.levelMax = 4096
        self.levelMin = 0
        self.name = name
        self.image = None
        self.axes = {}
        self.imageDisp = None
        self.ui = Ui_Form() #pyqtgraph.imageview.ImageViewTemplate_pyqt
        self.ui.setupUi(self)
        self.scene = self.ui.graphicsView.scene()
        
        self.ignoreTimeLine = False
        self.ignoreZLine = False
        
        if view is None:
            self.view = ViewBox()
        else:
            self.view = view
        
        self.ui.graphicsView.setCentralItem(self.view)
        self.view.setAspectLocked(True)
        self.view.invertY()
        
        if imageItem is None:
            self.imageItem = ImageItem()
        else:
            self.imageItem = imageItem
        self.imageview = self.view.addItem(self.imageItem)
        self.currentTime = 0
        self.currentLayer = 0 #layer in z axis
        
        #relay x y coordinates within image
        self.view.scene().sigMouseMoved.connect(self.mouseMoved)
        self.view.scene().sigMouseClicked.connect(self.mouseClicked)
        #self.view.scene().mouseReleaseEvent(self.releaseEvent)
        #self.view.scene().mouseReleaseEvent(self.mouseReleased)
        #pdb.set_trace()
        #generate roi list
        self.croi = [] #holdes x,y,z points for roi
        self.ccroi = [] #hold pg roi for croi
        self.aroi = {} #place to store all rois, each roi is passed as dictionary with each index = currentLayer
        self.aaroi = [] #place to store pg of arois in self.currentLayer
        self.button1 = 'off'
        
        self.ui.histogram.setImageItem(self.imageItem)
        
        self.menu = None
        
        self.ui.normGroup.hide()

        self.roi = PlotROI(10)
        self.roi.setZValue(20)
        self.view.addItem(self.roi)
        self.roi.hide()
        self.normRoi = PlotROI(10)
        self.normRoi.setPen('y')
        self.normRoi.setZValue(20)
        self.view.addItem(self.normRoi)
        self.normRoi.hide()
        self.roiCurve = self.ui.roiPlot.plot()
        
        
        self.timeLine = InfiniteLine(0, movable=True)
        self.timeLine.setPen((255, 255, 0, 200))
        #self.timeLine.setPen((255, 255, 200, 0))
        self.timeLine.setZValue(1)
        self.ui.roiPlot.addItem(self.timeLine)
        self.ui.splitter.setSizes([self.height()-35, 35])
        self.ui.roiPlot.hideAxis('left')
        
        
        
        self.ui.splitter.setSizes([self.height(), 10])
        self.ui.zPlot.plot()
        self.zLine = InfiniteLine(0, movable=True, angle = 0)
        #self.zLine.setBounds([0, 42])
        self.zLine.setPen((255, 255, 0, 200))
        self.zLine.setZValue(self.currentLayer )
        self.ui.ztext.setText(str(self.currentLayer))
        
        self.ui.zPlot.addItem(self.zLine)
        self.ui.zPlot.hideAxis('bottom')
        
        
        
        self.keysPressed = {}
        self.playTimer = QtCore.QTimer()
        self.playRate = 0
        self.lastPlayTime = 0
        
        self.normRgn = LinearRegionItem()
        self.normRgn.setZValue(0)
        self.ui.roiPlot.addItem(self.normRgn)
        self.normRgn.hide()
            
        ## wrap functions from view box
        for fn in ['addItem', 'removeItem']:
            setattr(self, fn, getattr(self.view, fn))

        ## wrap functions from histogram
        for fn in ['setHistogramRange', 'autoHistogramRange', 'getLookupTable', 'getLevels']:
            setattr(self, fn, getattr(self.ui.histogram, fn))

        self.timeLine.sigPositionChanged.connect(self.timeLineChanged)
        self.zLine.sigPositionChanged.connect(self.zLineChanged)
        self.ui.roiBtn.clicked.connect(self.roiClicked)
        self.roi.sigRegionChanged.connect(self.roiChanged)
        #self.ui.normBtn.toggled.connect(self.normToggled)
        self.ui.menuBtn.clicked.connect(self.menuClicked)
        self.ui.normDivideRadio.clicked.connect(self.normRadioChanged)
        self.ui.normSubtractRadio.clicked.connect(self.normRadioChanged)
        self.ui.normOffRadio.clicked.connect(self.normRadioChanged)
        self.ui.normROICheck.clicked.connect(self.updateNorm)
        self.ui.normFrameCheck.clicked.connect(self.updateNorm)
        self.ui.normTimeRangeCheck.clicked.connect(self.updateNorm)
        #radio button for navigation and roi selection
        self.ui.navRadio.clicked.connect(self.NavigationStatus)
        self.ui.radioSinglePoint.clicked.connect(self.radioSinglePoint)
        self.ui.radioAreaROI.clicked.connect(self.radioAreaROI)
        self.ui.radioPolygon.clicked.connect(self.radioPolygon)
        self.ui.radioEdit.clicked.connect(self.radioEdit)
        
        self.playTimer.timeout.connect(self.timeout)
        
        self.normProxy = SignalProxy(self.normRgn.sigRegionChanged, slot=self.updateNorm)
        self.normRoi.sigRegionChangeFinished.connect(self.updateNorm)
        
        self.ui.roiPlot.registerPlot(self.name + '_ROI')
        self.view.register(self.name)
        
        self.noRepeatKeys = [QtCore.Qt.Key_Right, QtCore.Qt.Key_Left, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down, QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown]
        
        self.roiClicked() ## initialize roi plot to correct shape / visibility

    
    def mouseMoved(self, pos):
        #print(evt[0])
        #print("Image position:", self.imageItem.mapFromScene(pos))
        #pdb.set_trace()

        #if self.mouseButton1 == 'on':
        #print('ButtonState:', QtCore.Qt.NoButton)
        print('Moving Mouse')
        #print(pos.button())
        #pdb.set_trace()
        #print(pos.button())
        #pos = pos.scenePos()

        if self.ui.radioAreaROI.isChecked():
            #if self.button1 == 'on':
            if self.view.scene().clickEvents:
                if self.view.scene().clickEvents[0].button() == 1:
                    print('Saving xy')
                    #pdb.set_trace()
                    if self.view .sceneBoundingRect().contains(pos):
                        self.clicked.append(self.view.scene().clickEvents)
                        mousePoint = self.view.mapSceneToView(pos)
                        index = int(mousePoint.x())
                        if index > 0 and index < self.image.shape[self.axes['x']]:
                            #label.setText("<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>,   <span style='color: green'>y2=%0.1f</span>" % (mousePoint.x(), data1[index], data2[index]))
                            #print(mousePoint.x())
                            #print(mousePoint.y())
                            #return(mousePoint.x(), mousePoint.y())
                            #pdb.set_trace()
                            if self.ccroi:
                                self.view.removeItem( self.ccroi) #removed roi generated from previous movements
                            #self.croi.append([mousePoint.x(), mousePoint.y(), self.currentLayer])#crois is the current roi being selected

                            self.croi.append([mousePoint.x(), mousePoint.y(), self.currentLayer])
                            self.ccroi = pg.PolyLineROI(self.croi, closed = False)
                            self.view.addItem( self.ccroi)  
                            print('appending roi coordinates')
                            print(self.croi) #crois is the current roi
                
            else:
                if self.croi:
                    if self.ccroi:
                        self.view.removeItem( self.ccroi) #remove the roi built has the mouse was dragged
                    #generate a novel key for dictionary holding rois
                    testname = 0
                    while testname in self.aroi.keys():
                        testname += 1
                        
                    self.aroi[testname] = {self.currentLayer : self.croi}
                    #pdb.set_trace()
                    self.aaroi.append(pg.PolyLineROI(self.croi, closed= True))
                    self.view.addItem(self.aaroi[-1])  
                    self.croi = []
                    self.ccroi =[]
                    print('transfered current roi (croi) to list for all rois (aroi)')     
                

    def mouseClicked(self, pos):
        #pdb.set_trace()
        print('mouseClicked')
        print(QtCore.Qt.NoButton)
        print(pos.button())
        #pdb.set_trace()
        self.view.scene()
        if self.ui.radioSinglePoint.isChecked():
            mousePoint = self.view.mapSceneToView(pos.scenePos())
            print('adding single point to roi list')
            #self.aroi.append([mousePoint.x(), mousePoint.y()])
            testname = 0
            while testname in self.aroi.keys():
                testname += 1
                        
            self.aroi[testname] = {self.currentLayer : [mousePoint.x(), mousePoint.y()]}
            self.aaroi.append(pg.CircleROI(self.circlePosition([mousePoint.x(), mousePoint.y()]), [self.markerRadius,self.markerRadius], pen = (4,9)))
            self.view.addItem(self.aaroi[-1])  
            
        
        if self.ui.radioPolygon.isChecked():
            mousePoint = self.view.mapSceneToView(pos.scenePos())
            if   pos.button()  ==   1:
                print('adding point to polygon')
                self.croi.append([mousePoint.x(), mousePoint.y()])
            elif pos.button() == 2:
                print('finished polygon and adding to roi list')
                self.aroi.append(self.croi.append([mousePoint.x(), mousePoint.y()]))
                self.croi = []
            



    def setImage(self, img, autoRange=True, autoLevels=True, levels=None, axes=None, xvals=None, pos=None, scale=None, transform=None, autoHistogramRange=True, image_Designation=None):
        """
        Set the image to be displayed in the widget.
        
        ================== ===========================================================================
        **Arguments:**
        img                (numpy array) the image to be displayed. See :func:`ImageItem.setImage` and
                           *notes* below.
        xvals              (numpy array) 1D array of z-axis values corresponding to the third axis
                           in a 3D image. For video, this array should contain the time of each frame.
        autoRange          (bool) whether to scale/pan the view to fit the image.
        autoLevels         (bool) whether to update the white/black levels to fit the image.
        levels             (min, max); the white and black level values to use.
        axes               Dictionary indicating the interpretation for each axis.
                           This is only needed to override the default guess. Format is::
                       
                               {'t':0, 'z':1, 'x':2, 'y':3, 'c':4};
        
        pos                Change the position of the displayed image
        scale              Change the scale of the displayed image
        transform          Set the transform of the displayed image. This option overrides *pos*
                           and *scale*.
        autoHistogramRange If True, the histogram y-range is automatically scaled to fit the
                           image data.
        ================== ===========================================================================

        **Notes:**        
        
        For backward compatibility, image data is assumed to be in column-major order (column, row).
        However, most image data is stored in row-major order (row, column) and will need to be
        transposed before calling setImage()::
        
            imageview.setImage(imagedata.T)nnn
            
        This requirement can be changed by the ``imageAxisOrder``
        :ref:`global configuration option <apiref_config>`.
        
        """
        self.image_Designation = image_Designation 
        profiler = debug.Profiler()
        
        if hasattr(img, 'implements') and img.implements('MetaArray'):
            img = img.asarray()
        
        if not isinstance(img, np.ndarray):
            required = ['dtype', 'max', 'min', 'ndim', 'shape', 'size']
            if not all([hasattr(img, attr) for attr in required]):
                raise TypeError("Image must be NumPy array or any object "
                                "that provides compatible attributes/methods:\n"
                                "  %s" % str(required))
        
        self.image = img
        self.imageDisp = None
        
        profiler()
        
        if axes is None:
            x,y = (0, 1) if self.imageItem.axisOrder == 'col-major' else (1, 0)
            
            if img.ndim == 2:
                self.axes = {'t': None, 'x': x, 'y': y, 'c': None}
            elif img.ndim == 3:
                # Ambiguous case; make a guess
                if img.shape[2] <= 4:
                    self.axes = {'t': None, 'z': None, 'x': x, 'y': y, 'c': 2}
                else:
                    self.axes = {'t': 0, 'z': None, 'x': x+1, 'y': y+1, 'c': None}
            elif img.ndim == 4:
                # Even more ambiguous; just assume the default
                if img.shape[2] <= 4:
                    self.axes = {'t': 0, 'z': None, 'x': x+2, 'y': y+2, 'c': 4}
                else: 
                    self.axes = {'t': 0, 'z': 1, 'x': x+2, 'y': y+2, 'c': None}
            elif img.ndim == 5:
                self.axes = {'t': 0, 'z': 1, 'x': x+2, 'y': y+2, 'c': 4}
            else:
                raise Exception("Can not interpret image with dimensions %s" % (str(img.shape)))
        elif isinstance(axes, dict):
            self.axes = axes.copy()
        elif isinstance(axes, list) or isinstance(axes, tuple):
            self.axes = {}
            for i in range(len(axes)):
                self.axes[axes[i]] = i
        else:
            raise Exception("Can not interpret axis specification %s. Must be like {'t': 2, 'x': 0, 'y': 1} or ('t', 'x', 'y', 'c')" % (str(axes)))
        
        for x in ['t', 'z', 'x', 'y', 'c']:
            self.axes[x] = self.axes.get(x, None)
        axes = self.axes
        
        #set min amd max values for time range
        if axes['t'] is not None:
            if hasattr(img, 'xvals'):
                try:
                    self.tVals = img.xvals(axes['t'])
                except:
                    self.tVals = np.arange(img.shape[axes['t']])
            else:
                self.tVals = np.arange(img.shape[axes['t']])
        
        #set min amd max values for z range
        if axes['z'] is not None:
            if hasattr(img, 'xvals'):
                try:
                    self.zVals = img.xvals(axes['z'])
                except:
                    self.zVals = np.arange(img.shape[axes['z']])
            else:
                self.zVals = np.arange(img.shape[axes['z']])

        
        profiler()
        
        self.currentTime = 0
        self.currentLayer =0
        self.updateImage(autoHistogramRange=autoHistogramRange)
        if levels is None and autoLevels:
            self.autoLevels()
        if levels is not None:  ## this does nothing since getProcessedImage sets these values again.
            self.setLevels(*levels)
          
        if self.ui.roiBtn.isChecked():
            self.roiChanged()

        profiler()
        #set max and min values for time 
        if self.axes['t'] is not None:
            #self.ui.roiPlot.show()
            self.ui.roiPlot.setXRange(self.tVals.min(), self.tVals.max())
            self.timeLine.setValue(0)
            #self.ui.roiPlot.setMouseEnabled(False, False)
            if len(self.tVals) > 1:
                start = self.tVals.min()
                stop = self.tVals.max() + abs(self.tVals[-1] - self.tVals[0]) * 0.02
            elif len(self.tVals) == 1:
                start = self.tVals[0] - 0.5
                stop = self.tVals[0] + 0.5
            else:
                start = 0
                stop = 1
            for s in [self.timeLine, self.normRgn]:
                s.setBounds([start, stop])
            #print(start)
            #print(stop)
        #else:
            #self.ui.roiPlot.hide()
            
                #set max and min values for time 
        if self.axes['z'] is not None:
            #self.ui.roiPlot.show()
            self.ui.zPlot.setYRange(self.zVals.min(), self.zVals.max())
            self.zLine.setValue(0)
            #self.ui.roiPlot.setMouseEnabled(False, False)
            if len(self.zVals) > 1:
                start = self.zVals.min()
                stop = self.zVals.max() + abs(self.zVals[-1] - self.zVals[0]) * 0.02
            elif len(self.zVals) == 1:
                start = self.zVals[0] - 0.5
                stop = self.zVals[0] + 0.5
            else:
                start = 0
                stop = 1
            for s in [self.zLine, self.normRgn]:
                s.setBounds([start, stop])
            print('zline value')
            print(self.zLine.value())
            print('start')
            print(start)
            print(stop)
        #else:
            #self.ui.roiPlot.hide()
        profiler()
        
        self.imageItem.resetTransform()
        if scale is not None:
            self.imageItem.scale(*scale)
        if pos is not None:
            self.imageItem.setPos(*pos)
        if transform is not None:
            self.imageItem.setTransform(transform)
        
        profiler()

        if autoRange:
            self.autoRange()
        self.roiClicked()

        profiler()
        self.view.setMouseEnabled(x=False, y=False) #turns panning and magnifcation off
        #pdb.set_trace() 
        
    def NavigationStatus(self):
        print('navigation button pressed')
        if self.ui.navRadio.isChecked():
            self.view.setMouseEnabled(x=True, y=True)
        else:
            self.view.setMouseEnabled(x=False, y=False)
            
    def radioSinglePoint(self):
        self.view.setMouseEnabled(x=False, y=False)
        
    def radioAreaROI(self):
        self.view.setMouseEnabled(x=False, y=False)
    
    def radioPolygon(self):
        self.view.setMouseEnabled(x=False, y=False)
    
    def radioEdit(self):
        self.view.setMouseEnabled(x=False, y=False)


    def clear(self):
        self.image = None
        self.imageItem.clear()
        
    def play(self, rate):
        """Begin automatically stepping frames forward at the given rate (in fps).
        This can also be accessed by pressing the spacebar."""
        #print "play:", rate
        self.playRate = rate
        if rate == 0:
            self.playTimer.stop()
            return
            
        self.lastPlayTime = ptime.time()
        if not self.playTimer.isActive():
            self.playTimer.start(16)

            
    def autoLevels(self):
        """Set the min/max intensity levels automatically to match the image data."""
        self.setLevels(self.levelMin, self.levelMax)


    def setLevels(self, min, max):
        """Set the min/max (bright and dark) levels."""
        self.ui.histogram.setLevels(min, max)

    def autoRange(self):
        """Auto scale and pan the view around the image such that the image fills the view."""
        image = self.getProcessedImage()
        self.view.autoRange()

        
    def getProcessedImage(self):
        """Returns the image data after it has been processed by any normalization options in use.
        This method also sets the attributes self.levelMin and self.levelMax 
        to indicate the range of data in the image."""
        if self.imageDisp is None:
            image = self.normalize(self.image)
            self.imageDisp = image
            self.levelMin, self.levelMax = list(map(float, self.quickMinMax(self.imageDisp)))
            
        return self.imageDisp

        
    def close(self):
        """Closes the widget nicely, making sure to clear the graphics scene and release memory."""
        self.ui.roiPlot.close()
        self.ui.graphicsView.close()
        self.scene.clear()
        del self.image
        del self.imageDisp
        super(ImageView, self).close()
        self.setParent(None)
    
    def mouseReleaseEvent(self, ev):
        print('mouseReleaseEvent')

        
    def keyPressEvent(self, ev):
        print('keyPressEvent')
        print(ev.key())
        #print ev.key()
        if ev.key() == QtCore.Qt.Key_Space:
            print('spacekey pressed')
            print('AllButtons:', QtCore.Qt.RightButton)
            #self.defineROI(ev)
            '''
            if self.playRate == 0:
                fps = (self.getProcessedImage().shape[0]-1) / (self.tVals[-1] - self.tVals[0])
                self.play(fps)
                #print fps
            else:
                self.play(0)
            ev.accept()
            '''
        elif ev.key() == QtCore.Qt.Key_Home:
            self.setCurrentIndex(0)
            self.play(0)
            ev.accept()
        elif ev.key() == QtCore.Qt.Key_End:
            self.setCurrentIndex(self.getProcessedImage().shape[0]-1)
            self.play(0)
            ev.accept()
        elif ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            self.keysPressed[ev.key()] = 1
            self.evalKeyState()
        else:
            QtGui.QWidget.keyPressEvent(self, ev)
    
    def defineROI(self, ev):
        print('startedROI')
        x=[]
        y=[]
        #pdb.set_trace()
        #pos = pos[0]  ## using signal proxy turns original arguments into a tuple
        while ev.key() == QtCore.Qt.Key_Space:
                    x.append(self.mouseMoved()[0])
                    y.append(self.mouseMoved()[1])
        print('all x y points:', x, y)
                

    def keyReleaseEvent(self, ev):
        if ev.key() in [QtCore.Qt.Key_Space, QtCore.Qt.Key_Home, QtCore.Qt.Key_End]:
            ev.accept()
        elif ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            try:
                del self.keysPressed[ev.key()]
            except:
                self.keysPressed = {}
            self.evalKeyState()
        else:
            QtGui.QWidget.keyReleaseEvent(self, ev)
        
    def evalKeyState(self):
        if len(self.keysPressed) == 1:
            key = list(self.keysPressed.keys())[0]
            if key == QtCore.Qt.Key_Right:
                self.play(20)
                self.jumpFrames(1)
                self.lastPlayTime = ptime.time() + 0.2  ## 2ms wait before start
                                                        ## This happens *after* jumpFrames, since it might take longer than 2ms
            elif key == QtCore.Qt.Key_Left:
                self.play(-20)
                self.jumpFrames(-1)
                self.lastPlayTime = ptime.time() + 0.2
            elif key == QtCore.Qt.Key_Up:
                self.play(-100)
            elif key == QtCore.Qt.Key_Down:
                self.play(100)
            elif key == QtCore.Qt.Key_PageUp:
                self.play(-1000)
            elif key == QtCore.Qt.Key_PageDown:
                self.play(1000)
        else:
            self.play(0)
        
    def timeout(self):
        now = ptime.time()
        dt = now - self.lastPlayTime
        if dt < 0:
            return
        n = int(self.playRate * dt)
        if n != 0:
            self.lastPlayTime += (float(n)/self.playRate)
            if self.currentTime+n > self.image.shape[0]:
                self.play(0)
            self.jumpFrames(n)
        
    def setCurrentIndex(self, ind):
        """Set the currently displayed frame index."""
        self.currentTime = np.clip(ind, 0, self.getProcessedImage().shape[self.axes['t']]-1)
        
        self.ignoreTimeLine = True
        self.timeLine.setValue(self.tVals[self.currentTime])
        self.ignoreTimeLine = False
        #self.currentLayer = np.clip(ind, 0, self.getProcessedImage().shape[self.axes['z']]-1)
        #self.zLine.setValue(self.zVals[self.currentLayer])
        self.updateImage()
        #self.ignoreZLine = False


    def jumpFrames(self, n):
        """Move video frame ahead n frames (may be negative)"""
        if self.axes['t'] is not None:
            self.setCurrentIndex(self.currentTime + n)


    def normRadioChanged(self):
        self.imageDisp = None
        self.updateImage()
        self.autoLevels()
        self.roiChanged()
        self.sigProcessingChanged.emit(self)
    
    def updateNorm(self):
        if self.ui.normTimeRangeCheck.isChecked():
            self.normRgn.show()
        else:
            self.normRgn.hide()
        
        if self.ui.normROICheck.isChecked():
            self.normRoi.show()
        else:
            self.normRoi.hide()
        
        if not self.ui.normOffRadio.isChecked():
            self.imageDisp = None
            self.updateImage()
            self.autoLevels()
            self.roiChanged()
            self.sigProcessingChanged.emit(self)

    def normToggled(self, b):
        self.ui.normGroup.setVisible(b)
        self.normRoi.setVisible(b and self.ui.normROICheck.isChecked())
        self.normRgn.setVisible(b and self.ui.normTimeRangeCheck.isChecked())

    def hasTimeAxis(self):
        return 't' in self.axes and self.axes['t'] is not None

    def roiClicked(self):
        showRoiPlot = False
        if self.ui.roiBtn.isChecked():
            showRoiPlot = True
            self.roi.show()
            #self.ui.roiPlot.show()
            self.ui.roiPlot.setMouseEnabled(True, True)
            self.ui.splitter.setSizes([self.height()*0.6, self.height()*0.4])
            self.roiCurve.show()
            self.roiChanged()
            self.ui.roiPlot.showAxis('left')
        else:
            self.roi.hide()
            self.ui.roiPlot.setMouseEnabled(False, False)
            self.roiCurve.hide()
            self.ui.roiPlot.hideAxis('left')
            
        if self.hasTimeAxis():
            showRoiPlot = True
            mn = self.tVals.min()
            mx = self.tVals.max()
            self.ui.roiPlot.setXRange(mn, mx, padding=0.01)
            self.timeLine.show()
            self.timeLine.setBounds([mn, mx])
            self.ui.roiPlot.show()
            if not self.ui.roiBtn.isChecked():
                self.ui.splitter.setSizes([self.height()-35, 35])
        else:
            self.timeLine.hide()
            #self.ui.roiPlot.hide()
            
        self.ui.roiPlot.setVisible(showRoiPlot)

    def roiChanged(self):
        if self.image is None:
            return
            
        image = self.getProcessedImage()
        if image.ndim == 2:
            axes = (0, 1)
        elif image.ndim == 3:
            axes = (1, 2)
        else:
            return
        
        data, coords = self.roi.getArrayRegion(image.view(np.ndarray), self.imageItem, axes, returnMappedCoords=True)
        if data is not None:
            while data.ndim > 1:
                data = data.mean(axis=1)
            if image.ndim == 3:
                self.roiCurve.setData(y=data, x=self.tVals)
            else:
                while coords.ndim > 2:
                    coords = coords[:,:,0]
                coords = coords - coords[:,0,np.newaxis]
                xvals = (coords**2).sum(axis=0) ** 0.5
                self.roiCurve.setData(y=data, x=xvals)

    def quickMinMax(self, data):
        """
        Estimate the min/max values of *data* by subsampling.
        """
        while data.size > 1e6:
            ax = np.argmax(data.shape)
            sl = [slice(None)] * data.ndim
            sl[ax] = slice(None, None, 2)
            data = data[sl]
        return nanmin(data), nanmax(data)


    def normalize(self, image):
        """
        Process *image* using the normalization options configured in the
        control panel.
        
        This can be repurposed to process any data through the same filter.
        """
        if self.ui.normOffRadio.isChecked():
            return image
            
        div = self.ui.normDivideRadio.isChecked()
        norm = image.view(np.ndarray).copy()
        #if div:
            #norm = ones(image.shape)
        #else:
            #norm = zeros(image.shape)
        if div:
            norm = norm.astype(np.float32)
            
        if self.ui.normTimeRangeCheck.isChecked() and image.ndim == 3:
            (sind, start) = self.timeIndex(self.normRgn.lines[0])
            (eind, end) = self.timeIndex(self.normRgn.lines[1])
            #print start, end, sind, eind
            n = image[sind:eind+1].mean(axis=0)
            n.shape = (1,) + n.shape
            if div:
                norm /= n
            else:
                norm -= n
                
        if self.ui.normFrameCheck.isChecked() and image.ndim == 3:
            n = image.mean(axis=1).mean(axis=1)
            n.shape = n.shape + (1, 1)
            if div:
                norm /= n
            else:
                norm -= n
            
        if self.ui.normROICheck.isChecked() and image.ndim == 3:
            n = self.normRoi.getArrayRegion(norm, self.imageItem, (1, 2)).mean(axis=1).mean(axis=1)
            n = n[:,np.newaxis,np.newaxis]
            #print start, end, sind, eind
            if div:
                norm /= n
            else:
                norm -= n
                
        return norm

        
    def timeLineChanged(self):
        #(ind, time) = self.timeIndex(self.ui.timeSlider)
        if self.ignoreTimeLine:
            return
        self.play(0)
        (ind, time) = self.timeIndex(self.timeLine)
        if ind != self.currentTime:
            self.currentTime = ind
            self.updateImage()
        #self.timeLine.setPos(time)
        #self.emit(QtCore.SIGNAL('timeChanged'), ind, time)
        self.sigTimeChanged.emit(ind, time)
        print('real time and ind')
        print(time)
        print(ind)
        
    def zLineChanged(self):
        print('zLineChanged')
        #(ind, time) = self.timeIndex(self.ui.timeSlider)
        if self.ignoreZLine:
            return
        (ind, time) = self.layerIndex(self.zLine)
        print('ind')
        print(ind)
        print('time')
        print(time)
        print('zLineValue')
        print(self.zLine.value())
        if ind != self.currentLayer:
            self.currentLayer = ind
            self.updateImage()
        #self.timeLine.setPos(time)
        #self.emit(QtCore.SIGNAL('timeChanged'), ind, time)
        self.sigTimeChanged.emit(ind, time)
        #remove old rois
        for croi in self.aaroi:
            self.view.removeItem( croi) 
        #change layer number in textbox
        self.ui.ztext.setText(str(self.currentLayer))
        #re-draw rois for the new layer
        for ckey in self.aroi.keys():
            croi = self.aroi[ckey]
            if self.currentLayer in croi.keys():
                #pdb.set_trace()
                if len(self.aroi[ckey][self.currentLayer]) == 2:
                    self.aaroi.append(pg.CircleROI(self.circlePosition(self.aroi[ckey][self.currentLayer]), [self.markerRadius, self.markerRadius])) 
                else:
                    self.aaroi.append(pg.PolyLineROI(self.aroi[ckey][self.currentLayer], closed= True))
                self.view.addItem(self.aaroi[-1])  
                
    def circlePosition(self, xy):
        x= xy[0]-self.markerRadius/2
        y=xy[1] - self.markerRadius/2
        return [x, y]

    def updateImage(self, autoHistogramRange=True):
        ## Redraw image on screen
        if self.image is None:
            return
            
        image = self.getProcessedImage()
        
        if autoHistogramRange:
            self.ui.histogram.setHistogramRange(self.levelMin, self.levelMax)
        
        # Transpose image into order expected by ImageItem
        if self.imageItem.axisOrder == 'col-major':
            axorder = ['t', 'z', 'x', 'y', 'c']
        else:
            axorder = ['t', 'z', 'y', 'x', 'c']
        axorder = [self.axes[ax] for ax in axorder if self.axes[ax] is not None]
        #pdb.set_trace()
        image = image.transpose(axorder)
            
        # Select time index
        if self.axes['t'] is not None:
            self.ui.roiPlot.show()
            image = image[self.currentTime, self.currentLayer]
        #pdb.set_trace()
        self.imageItem.updateImage(image)
            
            
    def timeIndex(self, slider):
        ## Return the time and frame index indicated by a slider
        if self.image is None:
            return (0,0)
        
        t = slider.value()
        
        xv = self.tVals
        if xv is None:
            ind = int(t)
        else:
            if len(xv) < 2:
                return (0,0)
            totTime = xv[-1] + (xv[-1]-xv[-2])
            inds = np.argwhere(xv < t)
            if len(inds) < 1:
                return (0,t)
            ind = inds[-1,0]
        return ind, t

    def layerIndex(self, slider):
        ## Return the time and frame index indicated by a slider
        if self.image is None:
            return (0,0)
        
        t = slider.value()
        
        xv = self.zVals
        if xv is None:
            ind = int(t)
        else:
            if len(xv) < 2:
                return (0,0)
            totTime = xv[-1] + (xv[-1]-xv[-2])
            inds = np.argwhere(xv < t)
            if len(inds) < 1:
                return (0,t)
            ind = inds[-1,0]
        return ind, t

    def getView(self):
        """Return the ViewBox (or other compatible object) which displays the ImageItem"""
        return self.view

        
    def getImageItem(self):
        """Return the ImageItem for this ImageView."""
        return self.imageItem

        
    def getRoiPlot(self):
        """Return the ROI PlotWidget for this ImageView"""
        return self.ui.roiPlot

       
    def getHistogramWidget(self):
        """Return the HistogramLUTWidget for this ImageView"""
        return self.ui.histogram


    def export(self, fileName):
        """
        Export data from the ImageView to a file, or to a stack of files if
        the data is 3D. Saving an image stack will result in index numbers
        being added to the file name. Images are saved as they would appear
        onscreen, with levels and lookup table applied.
        """
        img = self.getProcessedImage()
        if self.hasTimeAxis():
            base, ext = os.path.splitext(fileName)
            fmt = "%%s%%0%dd%%s" % int(np.log10(img.shape[0])+1)
            for i in range(img.shape[0]):
                self.imageItem.setImage(img[i], autoLevels=False)
                self.imageItem.save(fmt % (base, i, ext))
            self.updateImage()
        else:
            self.imageItem.save(fileName)

            
    def exportClicked(self):
        fileName = QtGui.QFileDialog.getSaveFileName()
        if fileName == '':
            return
        self.export(fileName)
        
    def buildMenu(self):
        self.menu = QtGui.QMenu()
        self.normAction = QtGui.QAction("Normalization", self.menu)
        self.normAction.setCheckable(True)
        self.normAction.toggled.connect(self.normToggled)
        self.menu.addAction(self.normAction)
        self.exportAction = QtGui.QAction("Export", self.menu)
        self.exportAction.triggered.connect(self.exportClicked)
        self.menu.addAction(self.exportAction)
        
    def menuClicked(self):
        if self.menu is None:
            self.buildMenu()
        self.menu.popup(QtGui.QCursor.pos())

    def setColorMap(self, colormap):
        """Set the color map. 

        ============= =========================================================
        **Arguments**
        colormap      (A ColorMap() instance) The ColorMap to use for coloring 
                      images.
        ============= =========================================================
        """
        self.ui.histogram.gradient.setColorMap(colormap)


    @addGradientListToDocstring()
    def setPredefinedGradient(self, name):
        """Set one of the gradients defined in :class:`GradientEditorItem <pyqtgraph.graphicsItems.GradientEditorItem>`.
        Currently available gradients are:   
        """
        self.ui.histogram.gradient.loadPreset(name)