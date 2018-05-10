'''
Created on May 11, 2017

@author: Surf32
'''
from pyqtgraph.Qt import QtGui
import os 
import pdb
import osDB

class ListSelection(QtGui.QDialog):
    def __init__(self, item_ls, parent=None):
        super(ListSelection, self).__init__(parent)
        self.result = ""
        #================================================= 
        # listbox
        #================================================= 
        self.listWidget = QtGui.QListWidget()
        for item in item_ls:
            w_item = QtGui.QListWidgetItem(item)
            self.listWidget.addItem(w_item)
        self.listWidget.itemClicked.connect(self.OnSingleClick)
        self.listWidget.itemActivated.connect(self.OnDoubleClick)
        layout = QtGui.QGridLayout()
        row=0
        layout.addWidget(self.listWidget,row,0,1,3) #col span=1, row span=3
        #================================================= 
        # OK, Cancel
        #================================================= 
        row +=1
        self.but_ok = QtGui.QPushButton("OK")
        layout.addWidget(self.but_ok ,row,1)
        self.but_ok.clicked.connect(self.OnOk)

        self.but_cancel = QtGui.QPushButton("Cancel")
        layout.addWidget(self.but_cancel ,row,2)
        self.but_cancel.clicked.connect(self.OnCancel)

        #================================================= 
        #
        #================================================= 
        self.setLayout(layout)
        self.setGeometry(300, 200, 460, 350)

    def OnSingleClick(self, item):
        self.result = item.text()

    def OnDoubleClick(self, item):
        self.result = item.text()
        self.close()
        return self.result

    def OnOk(self):
        if self.result == "":
            QMessageBox.information(self, "Error",
            "One item must be selected")
            return 
        self.close()
        return self.result

    def OnCancel(self):
        self.close()

    def GetValue(self):
        return self.result
    

class ImageChoiceResult():
    def __init__(self, r=True, g=True, b=True, pathToSecondaryImage = None, pathToMaskFile = None, loadtimeseries= True):
        self.r = r 
        self.b = b
        self.g = g 
        self.pathToSecondaryImage = pathToSecondaryImage
        self.pathToMaskFile = pathToMaskFile
        self.loadtimeseries = loadtimeseries
    
class ImageChoice(QtGui.QDialog):
    def __init__(self, workingdirectory = None, filename = None, parent=None):
        super().__init__()
        self.pathToSecondaryImage = None #path to secondary image if chosen
        self.workingdirectory = workingdirectory
        
        self.filename = filename
        layout = QtGui.QGridLayout()
        
        self.loadtimeseriesCheckBox = QtGui.QCheckBox('Load Time Series')
        self.loadtimeseriesCheckBox.setChecked(True)
        layout.addWidget(self.loadtimeseriesCheckBox, 0,0,1, 3)
        
        explainlabel = QtGui.QLabel('Select Channels to place image (green is plotted)')
        layout.addWidget(explainlabel, 1, 0, 1,3)
        
        self.r = QtGui.QCheckBox('red')
        layout.addWidget(self.r, 2, 0, 1,1)
        
        self.g = QtGui.QCheckBox('green')
        layout.addWidget(self.g, 2, 1, 1,1)
        
        self.b = QtGui.QCheckBox('blue')
        layout.addWidget(self.b, 2, 2, 1,1)

        self.image2btn = QtGui.QPushButton("Select Image for other channels")
        layout.addWidget(self.image2btn, 3, 0, 1,3)
        self.image2btn.clicked.connect(self.Load2ndImage)
        
        #search directory for npy files; will load if it iexists or save if does not exist
        file, index = osDB.getFileContString(self.workingdirectory, '.npy')
        if len(file) != 0:
            label1 = 'Load this numpy file?'
            self.pathToMaskFile = os.path.join(self.workingdirectory, file.values[0])
        else:
            label1 =  'Save mask in numpy file?'
            self.pathToMaskFile = os.path.join(self.workingdirectory, filename[:-4] + 'MaskNumpy.npy')
        explainlabel2 = QtGui.QLabel(label1)
        layout.addWidget(explainlabel2, 4, 0, 1,3)
        
        
        self.textBoxMaskFile = QtGui.QLineEdit(self.pathToMaskFile)
        layout.addWidget(self.textBoxMaskFile, 5, 0, 1, 3)
        self.textBoxMaskFile.textChanged.connect(self.textMaskChanged)
        
        self.continueBtn = QtGui.QPushButton("Settings Picked, Continue")
        layout.addWidget(self.continueBtn, 6, 0, 1, 3)
        self.continueBtn.clicked.connect(self.finished)
        
        self.setLayout(layout)
        self.setGeometry(300, 200, 460, 350)
        
    def finished(self):
        self.close()
        return self.result()
        
    def textMaskChanged(self, text):
        self.pathToMaskFile = text
    
    def Load2ndImage(self):
        print('Secondary Image start load')
        self. pathToSecondaryImage = QtGui.QFileDialog.getOpenFileName(self, 'Open Tif File', directory = self.workingdirectory)[0]
        
    def GetValue(self):
        return self.result()
        
    def result(self):
        return ImageChoiceResult(r=self.r.checkState(), g=self.g.checkState(), b=self.b.checkState(), pathToSecondaryImage=self.pathToSecondaryImage, pathToMaskFile=self.pathToMaskFile, loadtimeseries=self.loadtimeseriesCheckBox.checkState())
    

        
        
        
        
        
        
        
        
        