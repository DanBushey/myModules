'''
Created on May 10, 2017

@author: Surf32
designed to plot roi values
'''
from blaze import nan
import pandas as pd
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
from skimage.draw import polygon #used in roi creation

try:
    from bottleneck import nanmin, nanmax
except ImportError:
    from numpy import nanmin, nanmax

'''
class PlotROI(ROI):
    def __init__(self, size):
        ROI.__init__(self, pos=[0,0], size=size) #, scaleSnap=True, translateSnap=True)
        self.addScaleHandle([1, 1], [0, 0])
        self.addRotateHandle([0, 0], [0.5, 0.5])
'''

class ROIplotView(QtGui.QWidget):
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
    changePlot = QtCore.pyqtSignal(object)

    
    def __init__(self, parent=None, name=None, view=None, plotItem=None, *args):
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
        self.currentRowSelectedinTable = None
        self.markerRadius = 15 #circle radius marking single points/rois
        self.clicked = []
        self.image_Designation = None
        self.levelMax = 4096
        self.levelMin = 0
        self.name = name
        self.image = None
        self.axes = {}
        self.imageDisp = None
        self.ui = Ui_FormPlotROI() #pyqtgraph.imageview.ImageViewTemplate_pyqt
        self.ui.setupUi(self)
        self.scene = self.ui.plotwidget.scene()
        if view is None:
            self.view = ViewBox()
        else:
            self.view = view
        
        self.ui.plotwidget.setCentralItem(self.view)
        
        if plotItem is None:
            self.plotItem = PlotItem()
        else:
            self.plotItem = PlotItem
        
        self.plotview = self.view.addItem(self.plotItem)
        
        self.plotItem.updatePlot(np.random.normal(size =100))