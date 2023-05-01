import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import random
import os

def generate_matrix(config, size, distribution, distribution_range, replacements, prob_density, sample_choice, loc):

    if config == 'golfball':
        return np.random.choice([-20,-1,0,1,20], size=(size, size))
    elif config == 'normal':
        return np.random.beta(0.01, 0.01, size=(size, size))
    elif config == 'eigenfish':
        mat = np.array([[0, 0, -1, -1],
                        [1, -1, 0, 0], 
                        [0, 1, 0, 0],
                        [1, 0, -1, 0]], dtype=float)
        A = random.uniform(-3, 3)
        B = random.uniform(-3, 3)
        mat[2, 2] = A
        mat[3, 3] = B

        return mat

    elif config == 'random':

        mat = np.zeros((size, size), dtype=float)

        # Replace values in the matrix with values of either -1, 0, or 1 with 40% probability
        for i in range(size):
            for j in range(size):
                if random.random() < prob_density:
                    mat[i, j] = random.choice(sample_choice)

        for i in range(replacements):
            # Replace a random value in the matrix with a value from the distribution
            if distribution == 'uniform':
                A = random.uniform(-distribution_range, distribution_range)
            elif distribution == 'beta':
                A = np.random.beta(0.01, 0.01) * distribution_range
            elif distribution == 'laplace':
                A = np.random.laplace(0, 1) * distribution_range
            elif distribution == 'rayleigh':
                A = np.random.rayleigh(1) * distribution_range

            mat[loc[0]+i, loc[1]+i] = A

        return mat
            
    return mat

def plot_eigenvalues(matrix_size, config, num_matrices, cmap='bone'):

    if config == 'eigenfish':
        matrix_size = 4
    
    if config == 'random':
        distribution = random.choice(['uniform', 'beta', 'laplace','rayleigh'])
        distribution_range = np.random.randint(1, 10)
        replacements = random.randint(1, 10)
        matrix_size = random.randint(3, 10)
        prob_density = random.choice([0.1, 0.4, 0.7])
        sample_choice = random.choice([[-1, 0, 1], [-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3, 4, 5]])

        if replacements > matrix_size / 2:
            replacements = int(matrix_size / 2)

        loc = [np.random.randint(0,matrix_size - replacements), np.random.randint(0,matrix_size - replacements)]
        cmap = random.choice(['jet', 'bone', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'])

    L = np.zeros((num_matrices, matrix_size), dtype=np.complex128)

    for i in tqdm(range(num_matrices)):
        A = generate_matrix(config, matrix_size, distribution, distribution_range, replacements, prob_density, sample_choice, loc)
        L[i,:] = np.linalg.eigvals(A)
    L = L.flatten()

    H, x, y = np.histogram2d(L.imag,L.real,bins=1000)
    fig = plt.figure(figsize=(2,2), dpi = 200)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # remove white border
    plt.imshow(np.log(H+1),extent=[y[0], y[-1], x[0], x[-1]], cmap = cmap)

    os.makedirs('images', exist_ok=True)
    count = len([name for name in os.listdir('images') if name.endswith('.png')])
    plt.savefig(f'images/{count}_{config}_{matrix_size}_{cmap}.png',bbox_inches='tight', pad_inches=0, dpi = 10000)
    plt.close()

    # Write the matrix and config to a text file
    with open(f'images/{count}_{config}.txt', 'w') as f:
        f.write(f'config: {distribution}\n')
        f.write(f'distribution_range: {distribution_range}\n')
        f.write(f'num_matrices: {num_matrices}\n')
        f.write(f'matrix_size: {matrix_size}\n')
        f.write(f'cmap: {cmap}\n')
        f.write(f'example matrix: {A}\n')
        f.write(f'example eigenvalues: {L[:1000]}\n')
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--config', type=str, default='golfball')
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--num_matrices', type=int, default=10**6)
    parser.add_argument('--cmap', type=str, default=None)
    parser.add_argument('--n', type=int, default=1)
    args = parser.parse_args()  # Assign the parsed arguments to the 'args' variable

    for i in range(args.n):
        if args.random:
            if args.cmap is not None:
                cmap = args.cmap
            plot_eigenvalues(args.size, 'random', args.num_matrices, cmap=cmap)
        else:
            if args.cmap is not None:
                cmap = args.cmap
            plot_eigenvalues(args.size, args.config, args.num_matrices, cmap=args.cmap)