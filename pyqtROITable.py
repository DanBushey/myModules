'''
Created on May 9, 2017

@author: Surf32
'''
import numpy as np
from PyQt5 import  QtWidgets, QtGui, QtCore
import functools
import pdb


class roiTableWidget(QtWidgets.QTableWidget):
    '''
    classdocs
    '''
    newitemselected = QtCore.pyqtSignal(object)

    def __init__(self, pullDownMenu=None, graphscene=None, colors=None, *args, **kwds):
        '''
        Constructor
        '''
        QtWidgets.QTableWidget.__init__(self, *args)
        self.setRowCount(4)
        self.setColumnCount(1)
        self.setItem(0,0, QtWidgets.QTableWidgetItem('Test'))
       
        #self.itemPressed.connect(self.itemSelected)
        #self.cellPressed.connect(self.itemSelected)
        self.itemSelectionChanged.connect(self.itemSelected)
        self.graphscene = graphscene
        graphscene.change111.connect(self.changeROI) #detects change in roi from Timeseries data
        #create a pulldown menu
        self.pullDownMenu = pullDownMenu
        self.colors = colors
        self.actDict={}
        self.makeMenu()
        self.currentRow = None
        self.currentCol = None
        self.setData(graphscene.aroi)
        self.itemChanged.connect(self.changeName) # triggered if user inputs data into cell
        
    def changeName(self):
        print('Changing ROI name')
        #pdb.set_trace()
        if self.currentRow != None:
            cname = self.item(self.currentRow, self.currentColumn)
            print(cname.text())
            if self.graphscene.aroi.index[self.currentRow] != cname.text():
                self.graphscene.aroi.iloc[self.currentRow].rename(cname.text())
            if cname.text() not in self.pullDownMenu:
                print('newtype')
                print(type(cname.text()))
                self.pullDownMenu.append(cname.text())
                print('pulldownmenue', self.pullDownMenu)
                self.makeMenu()

    
    def makeMenu(self):
        self.remPullDownMenu()
        if self.currentColumn == 0:
            for item1 in self.pullDownMenu:
                #pdb.set_trace()
                actionEdit = QtGui.QAction(item1, self)
                actionEdit.triggered.connect(functools.partial(self.addItemAction, item1))
                self.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
                self.addAction(actionEdit)
                self.actDict[item1]=actionEdit
        elif self.currentColumn == 1:
            if self.colors != None:
                for item1 in self.pullDownMenu:
                    #pdb.set_trace()
                    actionEdit = QtGui.QAction(item1, self)
                    actionEdit.triggered.connect(functools.partial(self.addItemAction, item1))
                    self.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
                    self.addAction(actionEdit)
                    self.actDict[item1]=actionEdit
                    
    
    def addItemAction(self, i1):
        print('additem triggered')
        self.setItem(self.currentRow, self.currentColumn, QtGui.QTableWidgetItem(i1))
        #self.store[self.segmentation.columnnames[self.currentColumn]][self.currentRow]=i1
        
        
    def remPullDownMenu(self):
        for key in self.actDict:
            self.removeAction(self.actDict[key])
        
        
    def itemSelected(self):
        if self.currentRow != None:
            self.item(self.currentRow, self.currentColumn).setBackground(QtGui.QColor(255, 255, 255))
        if self.selectedItems():
            for currentTable in self.selectedItems():
                print('column', currentTable.column())
                print('row', currentTable.row())  
                self.currentColumn = currentTable.column()
                self.currentRow = currentTable.row()
                self.graphscene.currentRowSelectedinTable = currentTable.row()
            self.remPullDownMenu()
            self.makeMenu()
            self.item(self.currentRow, self.currentColumn).setBackground(QtGui.QColor(200, 0, 0))
        else:
            self.currentRow = None
            self.graphscene.currentRowSelectedinTable = None
            print('Nothing selected')
        self.newitemselected.emit('RoiChanged')
                
    def setData(self, data): 
        columns = ('ROI', 'Color', 'Type')
        self.setHorizontalHeaderLabels(columns)
        self.setColumnCount(len(columns))
        self.setRowCount(len(data))
        for i, roi_key in enumerate(data.index):
            if isinstance(roi_key, int):
                roi_key = str(roi_key)
            self.setItem(i, 0, QtWidgets.QTableWidgetItem(roi_key ))
            for i2, ccol in enumerate(columns[1:]):
                self.setItem(i, i2+1, QtWidgets.QTableWidgetItem(self.graphscene.aroi[ccol].loc[roi_key] ))
        self.resizeColumnsToContents()
    
    def changeROI(self):
        self.setData(self.graphscene.aroi)