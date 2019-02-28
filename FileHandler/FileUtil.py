__doc__ = 'description'
__author__ = '13314409603@163.com'

import os
import docx
import re
from win32com import client as wc
from enum import Enum



#获取txt文件类容
def getTXTContent(filePath):
    with open(filePath,encoding='UTF-8') as f:
        content = f.read()
        return f.read()

def getDOCContent(filePath):
    #doc文件转docx
    word = wc.Dispatch('Word.Application')
    doc = word.Documents.Open(filePath)
    docxName = filePath.replace('doc', 'docx')
    doc.SaveAs(docxName, 12)
    doc.Close()
    word.Quit()
    docxContent = docx.Document(docxName)
    paragraphs = []
    for parag in docxContent.paragraphs:
        paragraphs.append(parag.text)
    return paragraphs









if __name__=='__main__':
    b = BDFH ;
    len(BDFH)