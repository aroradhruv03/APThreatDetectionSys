# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import csv
write_flag = 'false'

path = '/Users/dhruv/Downloads/APT/SysAttack/sysattacker1'

def main():
    filePath = '/Users/dhruv/Downloads/APT/Debo/attack5/stream-1.txt'
    file = open(filePath,'r')
    
    """with file as f:
        for line in f:
            print line
            str = str+ line
            print str"""
            
    file_read = file.read()
    #file_read = file_read.decode('utf-8')
    
    #print file_read # for testing only

    fileCleaning(file_read)
    
       
def readFile(fileName):
    filePath = '/Users/dhruv/Downloads/APT/Shared/'+fileName
    file = open(filePath,'r')
    
    file_read = file.read()    
    
    #file_read = file_read.decode('utf-8')
    #print file_read # for testing only
    
    fileCleaning(file_read,fileName)


def loadCSV(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


    
def fileCleaning(file_object):
    tokenized = tokenize(file_object)
    print tokenized
    
def tokenize(file_obj):
    from nltk.tokenize import word_tokenize
    
    tokenized = word_tokenize(file_obj)
    writeFile(tokenized,"/Users/dhruv/Downloads/APT/Debo/attack5/stream-1.txt")
    print tokenized

def cleanPunc(tokenized):
    
    import re
    import string
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    
    
    # print regex.findall # for testing only
    
    tokenized_no_punc = []
    tokenized_no_punc2 = []
    
    for token in tokenized: 
            new_token = regex.sub(u'', token)
            if not new_token == u'':
                tokenized_no_punc.append(new_token)
    
    regex = re.compile(r'[-.?!,":;()|0-9]')
    for token in tokenized_no_punc: 
            new_token = regex.sub(u'', token)
            if not new_token == u'':
                tokenized_no_punc2.append(new_token)
    
                
    return tokenized_no_punc2
    # print "No Punctuation\n",tokenized_no_punc,"\n"
    
def writeFile(final_cleaned_text,fileName):
    global write_flag
    
    if(write_flag=='true'):
        #target = open('/Users/Dhruv/Mobuyle/Output/'+fileName, 'w')
        target = open('/Users/dhruv/Downloads/APT/Debo/attack5/stream-all.txt', 'w')
        for word in final_cleaned_text:
            word = word.encode('utf-8')
            target.write(word)
            target.write(' ')
        target.close()
        write_flag='false'
    else:
        #target = open('/Users/Dhruv/Mobuyle/Output/'+fileName, 'a')
        target = open('/Users/dhruv/Downloads/APT/Debo/attack5/stream-all.txt', 'a')
        target.write('Attack \n')
        for word in final_cleaned_text:
            word = word.encode('utf-8')
            target.write(word)
            target.write(' ')
        target.close()
    
if __name__== '__main__':
    main()