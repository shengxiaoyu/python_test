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
    path = r'E:\研二1\学术论文\准备材料2\离婚纠纷第二批（分庭审笔录）\含庭审笔录\(2017)津0104民初9121号\庭审笔录\9121李双庆第一次开庭笔录.doc'
    print(getDOCContent(path))