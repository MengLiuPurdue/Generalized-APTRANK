[![Binder](http://mybinder.org/badge.svg)](http://beta.mybinder.org/v2/gh/MengLiuPurdue/Generalized-APTRANK/master)

## Intro
This is a generalized APTRANK based on [Jiang, Biaobin, et al. "AptRank: An Adaptive PageRank Model for Protein Function Prediction on Bi-relational Graphs." arXiv preprint arXiv:1601.05506 (2016).](https://arxiv.org/abs/1601.05506)

## Input

* Input:
* ei, ej       - an edge list
* m, n         - the number of rows and edges of the original data set (to ensure
               the reconstructed matrix has the same dimension of the original one)
* train_rows   - the row positions of training data, represented in a tuple of arrays
* train_cols   - the column positions of training data, represented in a tuple of arrays
* predict_rows - the row positions where you want predictions
* predict_cols - the column positions where you want predictions
* K            - the number of terms used in the APTRANK, default is 8
* S            - the number of independent experiments during training,
               default is 5
* diff_type    - choose what kind of diffusion matrix you want, 1 for G/rho, 2 for
               G*D^{-1}, 3 for (D - G)/rho and 4 for (I - D^{-1/2}*G*D^{-1/2}),
               where rho is the spectral radius and D is the out-degree of each
               vertex, defalut is 1
* ratio        - the split ratio between fiiting set and validation set, default is 0.8
* rho          - a parameter used to control the diffusion process, rho > 0, if a
                rho <= 0 is given, it will be replaced by the spectral radius
* seeds        - seed positions, where the diffusion starts, default is the same as predict_cols
* sampling_type - 1 for randomly sampling using ratio, 2 for S-fold cross validation, default is 1

## Output

* Xa            - the predicted matrix, zero rows or columns in the original graph will remain zero.
* alpha         - the final parameters of APTRANK
* all_alpha     - records of alphas in each independent experiment

## Example
For a quick sample run:

```
include("test.jl")
```
 


