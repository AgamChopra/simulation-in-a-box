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
    sequence = list(gene)
    sequence = encodings[sequence[0]]+encodings[sequence[1]]+encodings[sequence[2]]+encodings[sequence[3]
                                                                                              ]+encodings[sequence[4]]+encodings[sequence[5]]+encodings[sequence[6]]+encodings[sequence[7]]
    return sequence


def mutate(mutation_factor, sequence):
    sequence = list(sequence)
    if mutation_factor > random.random():
        i = random.randint(0, len(sequence)-1)
        if sequence[i] == '0':
            sequence[i] = '1'
        else:
            sequence[i] = '0'
    sequence = ''.join(sequence)
    return sequence


def tangle(sequence):
    gene = []
    [gene.append(decodings[sequence[i:i+5]])
     for i in range(0, len(sequence), 5)]
    gene = ''.join(gene)
    return gene


def split_seq(utg):
    source = utg[0]  # input or hidden
    source_id = utg[1:8]  # address of either input or hidden
    sink_type = utg[8]  # sink/aka the output. output neuron or hidden neuron
    sink_id = utg[9:16]  # id of output neuron or output hidden neuron
    recurrent = utg[16]  # if the neuron has memory
    # value of weight first bit represents the sign(0:+ve,1:-ve)
    weight = utg[17:]
    return source, source_id, sink_type, sink_id, recurrent, weight


gene = 'AX0W1ZXX' #<- how gene is stored(in memory) and displayed(to user)
print(gene, untangle(gene))
for i in range(5000):
    utg = untangle(gene) #<- gene is untangled to a binary sequence to be used by the cell
    utg = mutate(global_mutation, utg) #<- during reproduction, there is a small chance(global_mutation factor) that a bit gets flipped in the untangled binary gene sequence.
    gene = tangle(utg) #<- After reproduction, the gene is tangled and stored in memory.
    if((i+1) % 50 == 0):
        print(gene, untangle(gene))
    # print(split_seq(utg))
