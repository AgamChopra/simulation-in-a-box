import random

global_mutation = 0.002

encodings = {'0': '00000', '1': '00001', '2': '00010', '3': '00011', '4': '00100',
             '5': '00101', 'A': '00110', 'B': '00111', 'C': '01000', 'D': '01001',
             'E': '01010', 'F': '01011', 'G': '01100', 'H': '01101', 'I': '01110',
             'J': '01111', 'K': '10000', 'L': '10001', 'M': '10010', 'N': '10011',
             'O': '10100', 'P': '10101', 'Q': '10110', 'R': '10111', 'S': '11000',
             'T': '11001', 'U': '11010', 'V': '11011', 'W': '11100', 'X': '11101',
             'Y': '11110', 'Z': '11111'}

decodings = dict([(value, key) for key, value in encodings.items()])


def untangle(gene):
    sequence = ''
    for char in gene:
        sequence += encodings[char] 
    return sequence


def mutate(sequence):
    sequence = list(sequence)
    if global_mutation > random.random():
        i = random.randrange(0, 40)
        sequence[i] = '1' if sequence[i] == '0' else '0'
    return ''.join(sequence)


def tangle(sequence):
    gene = []
    [gene.append(decodings[sequence[i:i+5]])
     for i in range(0, len(sequence), 5)]
    return ''.join(gene)

def bin_to_float(string):
    num1 = sum([int(string[1 + i]) * 2 ** (10 - i) for i in range(11)])
    num2 = sum([int(string[12 + i]) * 2 ** -(1 + i) for i in range(0,11)])
    return num1 + num2 if string[0] == '0' else -(num1 + num2)

def split_seq(utg):
    source = utg[0]  # input or hidden
    source_id = utg[1:8]  # address of either input or hidden
    sink_type = utg[8]  # sink/aka the output. output neuron or hidden neuron
    sink_id = utg[9:16]  # id of output neuron or output hidden neuron
    recurrent = utg[16]  # if the neuron has memory
    weight = utg[17:]# value of weight's first bit represents the sign(0:+ve,1:-ve) # weight = [sign] [11 bits] [.] [11 bits]. ex- 1 11111111111 . 11111111111 -> -2047.99951171875
    ##lr = utg[40:] sequence of 5 bits
    return source, source_id, sink_type, sink_id, recurrent, weight


gene = 'AX0W1ZXX' #<- how gene is stored(in memory) and displayed(to user)
print(gene, untangle(gene))
for i in range(5000):
    utg = untangle(gene) #<- gene is untangled to a binary sequence to be used by the cell
    utg = mutate(utg) #<- during reproduction, there is a small chance(global_mutation factor) that a bit gets flipped in the untangled binary gene sequence.
    gene = tangle(utg) #<- After reproduction, the gene is tangled and stored in memory.
    if((i+1) % 50 == 0):
        print(gene, untangle(gene))
        source, source_id, sink_type, sink_id, recurrent, weight = split_seq(utg)
        print(source, source_id, sink_type, sink_id, recurrent, weight)
        print('weight:', bin_to_float(weight))