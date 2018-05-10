'''
Created on Feb 5, 2017

@author: Surf32
'''
import pandas as pd

def cleanXlsx(inputfile, ouputfile):
    reader = pd.ExcelFile(inputfile)
    writer = pd.ExcelWriter(ouputfile)
    for sheet in reader.sheet_names:
        sheet1 = pd.read_excel(ExcelFile, sheetname = sheet, encoding='utf-8')
        sheet1.to_excel(writer, sheet_name = sheet)
    writer.save()