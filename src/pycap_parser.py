# -*- coding: utf-8 -*-
"""
Created on Sun May  8 01:16:46 2016

@author: dhruv
"""

import pcapy
from os import listdir
import json

benign_packets = []
attack_packets = []
packets = []
i=0
write_flag='false'

def main():
    benign_files = [f for f in listdir("/Users/dhruv/Downloads/APT/Debo/benign_temp")]
    attack_files =  [f for f in listdir("/Users/dhruv/Downloads/APT/Debo/attack_temp")]
    print attack_files
    readData(attack_files)
    readData(benign_files)
    
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
    
def readData(folder_name) :   
    for fileName in folder_name:
        if fileName==".DS_Store":
            continue
        if fileName.endswith('.cap') or fileName.endswith('.pcap'):            
            print fileName
            reader = pcapy.open_offline('/Users/dhruv/Downloads/APT/Debo/attack_temp/'+fileName)
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
        writeFile(s,"attack")
        
        
def writeFile(data,typeAttack):
    global i
    global write_flag
    
    target = open('/Users/dhruv/Downloads/APT/Debo/train.tsv', 'a')
    data = data.encode('utf-8')
    
    if(write_flag=='false'):
        write_flag='true'
        target.write("id\ttype\tdata\n")
        
        
        
    #target = open('/Users/Dhruv/Mobuyle/Output/'+fileName, 'a')
    
    target.write("\""+str(i)+"\"\t")
    target.write('attack\t')
    target.write(data)
    target.write('\n')
    target.close()
    i=i+1

if __name__== '__main__':
    main()