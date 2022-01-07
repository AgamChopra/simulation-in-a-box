

import torch

NUM_START = 3
LOW = 100
HIGH = 800

encodings = {'0': '00000', '1': '00001', '2': '00010', '3': '00011', '4': '00100',
             '5': '00101', 'A': '00110', 'B': '00111', 'C': '01000', 'D': '01001',
             'E': '01010', 'F': '01011', 'G': '01100', 'H': '01101', 'I': '01110',
             'J': '01111', 'K': '10000', 'L': '10001', 'M': '10010', 'N': '10011',
             'O': '10100', 'P': '10101', 'Q': '10110', 'R': '10111', 'S': '11000',
             'T': '11001', 'U': '11010', 'V': '11011', 'W': '11100', 'X': '11101',
             'Y': '11110', 'Z': '11111'}

idx_live = torch.ones(NUM_START)

idx_pos = torch.randint(LOW, HIGH, (NUM_START,2))

idx_gene_list = [['AX0W1ZXX','AX0W1ZXX'],['AX0W1ZXX','AX0W1ZXX'],['AX0W1ZXX','AX0W1ZXX']]

print(idx_live.shape, idx_pos.shape, idx_gene_list)

#%%

WIDTH,HEIGHT = 1600, 900

arty = torch.zeros((WIDTH,HEIGHT,1),dtype=torch.int32)
arty[10,3] = 230150128

r = int(arty[10,3]/1E6)
g = int((arty[10,3]- r*1E6)/1E3) 
b = int(arty[10,3] - r*1E6 - g*1E3)

print(r,g,b)
