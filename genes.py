import random
from tqdm import trange
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

global_mutation = 0.005

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


def mutate(sequence, mutation_rate=0.1):
    sequence = list(sequence)
    # At least 10% or 1 mutation
    num_mutations = max(1, int(len(sequence) * mutation_rate))
    indices_to_mutate = random.sample(range(len(sequence)), num_mutations)

    for i in indices_to_mutate:
        if random.random() <= global_mutation:
            sequence[i] = '1' if sequence[i] == '0' else '0'

    return ''.join(sequence)


def tangle(sequence):
    gene = []
    [gene.append(decodings[sequence[i:i+5]])
     for i in range(0, len(sequence), 5)]
    return ''.join(gene)


def bin_to_float(string):
    num1 = sum([int(string[1 + i]) * 2 ** (10 - i) for i in range(11)])
    num2 = sum([int(string[12 + i]) * 2 ** -(1 + i) for i in range(0, 11)])
    return num1 + num2 if string[0] == '0' else -(num1 + num2)


def create_neural_weights(utg):
    neural_weights = []
    for i in range(100):
        start = 216 + i * 76  # Starting index for each set
        slice1 = utg[start:start + 23]
        slice2 = utg[start + 23:start + 46]
        slice3 = utg[start + 46:start + 69]
        charge_time = utg[start + 69:start + 74]
        self_firing = utg[start + 74]
        plastic = utg[start + 75]
        neural_weights.append(
            [slice1, slice2, slice3, charge_time, self_firing, plastic])
    return neural_weights


def split_seq(utg):
    eye = [utg[0], utg[1:24]]
    thermal = [utg[24], utg[25:48]]
    photo = [utg[48], utg[49:72]]
    tactile = [utg[72], utg[73:96]]

    velo_add = [utg[96], utg[97:120]]
    orient = [utg[120], utg[121:144]]
    kill = [utg[144], utg[145:168]]
    eat = [utg[168], utg[169:192]]
    reproduce = [utg[192], utg[193:216]]
    neural_weights = create_neural_weights(utg)

    return eye, thermal, photo, tactile, velo_add, orient, kill, eat, reproduce, neural_weights


def mutation_guarding_gene(N=4):
    gene = []
    [gene.append('0') for _ in range(N)]
    return ''.join(gene)


def generate_random_string(length):
    characters = list(encodings.keys())
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string


def calculate_similarity(segment):
    return sum(int(bit) for bit in segment) / len(segment)


def assign_color(eye, thermal, photo, tactile, velo_add, orient, kill, eat, reproduce):
    green_segments = thermal[1] + photo[1]
    red_segments = kill[1] + eat[1]
    blue_segments = eye[1] + velo_add[1] + orient[1] + tactile[1]

    green_alph = calculate_similarity(green_segments)
    red_alph = calculate_similarity(red_segments)
    blue_alph = calculate_similarity(blue_segments)

    green = int((int(thermal[0]) * 127 + int(photo[0]) * 127) * green_alph)
    red = int((int(kill[0]) * 127 + int(eat[0]) * 127) * red_alph)
    blue = int((int(eye[0]) * 63 + int(velo_add[0]) * 63 + int(orient[0])
               * 63 + int(tactile[0]) * 63) * blue_alph)

    luminosity = int(reproduce[0], 2) % 2 == 1

    return (red, green, blue, luminosity)


def main():
    required_length = 216 + 100 * 76
    gene = generate_random_string(required_length)

    red_values = []
    green_values = []
    blue_values = []
    luminosity_values = []

    for i in trange(10000):
        utg = untangle(gene)
        utg = mutate(utg)
        gene = tangle(utg)

        if ((i+1) % 1 == 0):
            eye, thermal, photo, tactile, velo_add, orient, kill, eat, reproduce, neural_weights = split_seq(
                utg)
            # print(gene)
            # print('eye:', eye[0], bin_to_float(eye[1]))
            color = assign_color(eye, thermal, photo, tactile,
                                 velo_add, orient, kill, eat, reproduce)
            red, green, blue, luminosity = color
            red_values.append(red)
            green_values.append(green)
            blue_values.append(blue)
            luminosity_values.append(luminosity)

    # Plotting the RGB values over iterations
    plt.figure(figsize=(15, 5))
    plt.plot(red_values, label='Red', color='red')
    plt.plot(green_values, label='Green', color='green')
    plt.plot(blue_values, label='Blue', color='blue')
    plt.xlabel('Iterations (in multiples of 50)')
    plt.ylabel('Color Intensity (0-255)')
    plt.title('Color Intensity over Iterations')
    plt.legend()
    plt.show()
    
    fig, ax = plt.subplots(figsize=(15, 5))
    points = np.array([range(len(red_values)), range(len(red_values))]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    colors = np.array([list(color) for color in zip(red_values, green_values, blue_values)]) / 255
    
    lc = LineCollection(segments, colors=colors, linewidth=2)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_xlim(0, len(red_values))
    ax.set_ylim(0, 1)
    ax.set_xlabel('Iterations (in multiples of 50)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('Color Intensity over Iterations')
    plt.show()


if __name__ == '__main__':
    main()
