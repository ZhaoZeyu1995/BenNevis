#!/usr/bin/env python3

# Apache 2.0


'''
0 -> a0:a -> (a1:<eps>) -> 0    Note: () represents self-loop
          -> - - - - - - > 0
'''

import sys
import logging

fread = open(sys.argv[1], 'r')
if len(sys.argv) > 2 and sys.argv[2] == 'no_blk':
    no_blk = True # True or False
else:
    no_blk = False

nodeX = 1

if not no_blk:
    print(str(0) + ' ' + str(0) + ' ' + '<blk>' + ' ' + '<eps>') # <blk> self-loop on state 0

for entry in fread.readlines():
    entry = entry.replace('\n', '').strip()
    fields = entry.split(' ')
    phone = fields[0]
    if not no_blk:
        if phone == '<eps>' or phone == '<SIL>':
            continue
    else:
        if phone == '<eps>':
            continue
        elif phone == '<SIL>':
            print(str(0) + ' ' + str(nodeX) + ' ' + '<SIL>' + ' ' + '<SIL>') 
            print(str(nodeX) + ' ' + str(nodeX) + ' ' + '<SIL>' + ' ' + '<eps>') # self-loop on state 1
            print(str(nodeX) + ' ' + str(0) + ' ' + '<eps>' + ' ' + '<eps>')
            nodeX += 1
            continue
    if phone.startswith('#'):
        print(str(0) + ' ' + str(0) + ' ' + phone + ' ' + phone)
    else:
        print(str(0) + ' ' + str(nodeX) + ' ' + phone+'_0' + ' ' + phone) # transiting the first state
        print(str(nodeX) + ' ' + str(0) + ' ' + '<eps> <eps>') # transiting back to the initial state
        print(str(nodeX) + ' ' + str(nodeX+1) + ' ' + phone+'_1' + ' <eps>') #  transiting to the second state
        print(str(nodeX+1) + ' ' + str(nodeX+1) + ' ' + phone+'_1' + ' <eps>') #  transiting to the second state
        print(str(nodeX+1) + ' ' + str(0) + ' ' + '<eps> <eps>') # transiting back to the initial state
        nodeX += 2

print('0')


fread.close()
