# -*- coding: utf-8 -*-
"""
Created on Sun May  8 01:16:46 2016

@author: dhruv
@version: 2.0
"""

import pcapy
from os import listdir
import json


i=0
write_flag='false'

INPUT_PATH_ATTACK = "/Users/dhruv/Downloads/APT/InputData/netattacker"
INPUT_PATH_BENIGN = "/Users/dhruv/Downloads/APT/InputData/netbenign"
OUTPUT_PATH_PARSED = "/Users/dhruv/Downloads/APT/InputData/"
TRAINING_FILE = "train.tsv"
TESTING_FILE = "test.tsv"

RANDOM_INPUT = '/Users/dhruv/Downloads/APT/InputData/RandomInputs'
RANDOM_INP_FILE = 'random_test.tsv'
RANDOM_INPUT_ATCK = '/Users/dhruv/Downloads/APT/InputData/RandomInputsAtck'

def main():
    global i
    global write_flag
    readFileNames(INPUT_PATH_ATTACK,"attack",TRAINING_FILE)
    readFileNames(INPUT_PATH_BENIGN,"benign",TRAINING_FILE)

    write_flag = 'false'
    i=0
    readFileNames(RANDOM_INPUT,"benign",RANDOM_INP_FILE)
    readFileNames(RANDOM_INPUT_ATCK,"attack",RANDOM_INP_FILE)




def readFileNames(input_path,typeAttack,dest_fileName):
    from os import walk

    f = []
    for (dirpath, dirnames, filenames) in walk(input_path):
        f.extend(filenames)
        break
    print dirnames
    #print filenames

    for dir_Names in dirnames:
        attack_folder = [fi for fi in listdir(input_path+"/"+dir_Names)]
        print attack_folder
        readData(input_path,dir_Names,attack_folder,typeAttack,dest_fileName)

    """
    cap = pcapy.open_offline('/Users/dhruv/Downloads/APT/Debo/attack_temp/stream-1.cap')
    s= ""
    (header, payload) = cap.next()
    while header:
        lenP = header.getlen()
        print ('cap', lenP)
        lengthString = str(lenP)
        s += lengthString+" "
        (header, payload) = cap.next()

    data = "\t\t"+s
    print data
    """

def readData(input_path,dir_Names,folder_name,typeAttack,dest_fileName) :
    for fileName in folder_name:
        if fileName==".DS_Store":
            continue
        if fileName.endswith('.cap') or fileName.endswith('.pcap'):
            print fileName
            reader = pcapy.open_offline(input_path+'/'+dir_Names+'/'+fileName)
            s= ""
            (header,length) = reader.next()
            while header:
                try:
                    length = header.getlen()
                    lengthString = str(length)
                    s += lengthString+" "
                    (header,length) = reader.next()
                except pcapy.PcapError:
                    continue
            writeFile(s,typeAttack,dest_fileName)


def writeFile(data,typeAttack,dest_fileName):
    global i
    global write_flag

    target = open(OUTPUT_PATH_PARSED+'/'+dest_fileName, 'a')
    data = data.encode('utf-8')

    if(write_flag=='false'):
        write_flag='true'
        target.write("id\ttype\tdata\n")



    #target = open('/Users/Dhruv/Mobuyle/Output/'+fileName, 'a')

    target.write("\""+str(i)+"\"\t")
    target.write(typeAttack+'\t')
    target.write(data)
    target.write('\n')
    target.close()
    i=i+1

if __name__== '__main__':
    main()