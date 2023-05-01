# bohemian_matrices
A place for experimenting with Bohemian Matrices.

## ?

This code generates and visualizes the eigenvalues of Bohemian matrices. Bohemian matrices are matrices with entries from a discrete set of integers. The visualization of their eigenvalues often creates beautiful patterns and, many people don't know this, but they hold the key to the universe. More here: http://www.bohemianmatrices.com/.

## Usage

You can run the program from the command line with the following arguments:

- __--random__: If provided, the code will generate a random Bohemian matrix.
- __--config__ : The type of matrix to generate. Options are `golfball`, `normal`, `eigenfish`, and `random`. Default is `golfball`.
- __--size__: The size of the matrix to generate. Default is 5.
- __--num\_matrices__: The number of matrices to generate. Default is $10^6$.
- __--cmap__: The colormap to use for the visualization. If not provided, the default colormap is used.
- __--n__: The number of iterations to run. Default is 1.

For example, to generate a `golfball` matrix of size 5, you would use the following command:

```bash
python main.py --config golfball --size 5
```

And to generate a random matrix, you would use the following command:

```bash
python main.py --random
```

## Output

The output of the program is a .png image of the eigenvalue visualization, saved in the `images` directory. The filename includes the configuration, the matrix size, and the colormap used.

Along with the .png image, the program also saves a .txt file containing detailed information about the matrix configuration, the number of matrices, the size of the matrix, the colormap used, and examples of the generated matrix and its eigenvalues. The .txt file is saved in the `images` directory and shares the same base name as the .png image file.
