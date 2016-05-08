# -*- coding: utf-8 -*-
"""
Created on Sun May  8 01:16:46 2016

@author: dhruv
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

def main():

    #benign_folder1 = [f for f in listdir("/Users/dhruv/Downloads/APT/Debo/benign_temp")]
    #attack_folder1 =  [f for f in listdir("/Users/dhruv/Downloads/APT/Input Data/netattackermay3")]

    #print attack_folder1
    #readData(attack_folder1)
    #readData(benign_folder1)

    readFileNames(INPUT_PATH_ATTACK,"attack")
    readFileNames(INPUT_PATH_BENIGN,"benign")


def readFileNames(input_path,typeAttack):
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
        readData(input_path,dir_Names,attack_folder,typeAttack)

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

def readData(input_path,dir_Names,folder_name,typeAttack) :
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
            writeFile(s,typeAttack)


def writeFile(data,typeAttack):
    global i
    global write_flag

    target = open(OUTPUT_PATH_PARSED+'/'+TRAINING_FILE, 'a')
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