# Recommender System Using Matrix Factorization

This repo performs matrix factorization using alternating least squares minimization to provide book recommendations.
The implementation is based on the
paper [Matrix Factorization Techniques for Recommender Systems](https://ieeexplore.ieee.org/document/5197422)
and draws on the
corresponding [CMU 10-315 course homework assignment](https://www.cs.cmu.edu/~10315/assignments/hw10/programming/). The
data used in this project is a Goodreads book ratings dataset from UCSD Book Graph. The original dataset authors' papers
are cited below.

## Citations

Y. Koren, R. Bell and C.
Volinsky, "[Matrix Factorization Techniques for Recommender Systems](https://ieeexplore.ieee.org/document/5197422)," in
Computer, vol. 42, no. 8, pp. 30-37, Aug. 2009, doi: 10.1109/MC.2009.263.

Mengting Wan, Julian
McAuley, "[Item Recommendation on Monotonic Behavior Chains](https://www.google.com/url?q=https%3A%2F%2Fgithub.com%2FMengtingWan%2Fmengtingwan.github.io%2Fraw%2Fmaster%2Fpaper%2Frecsys18_mwan.pdf&sa=D&sntz=1&usg=AFQjCNGGcNRW1tSZKPWO0yZsr8mj7MkWuw)"
, in RecSys'18.

Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian
McAuley, "[Fine-Grained Spoiler Detection from Large-Scale Review Corpora](https://www.google.com/url?q=https%3A%2F%2Fwww.aclweb.org%2Fanthology%2FP19-1248&sa=D&sntz=1&usg=AFQjCNG8xlMi09lyuzzMI8lCW58wrBEGsQ)"
, in ACL'19.

## Future Work

- [ ] Add bias consideration to matrix factorization
- [ ] Regularize loss function
- [ ] Expand dataset to include all genres
- [ ] Add frontend for getting specific recommendations

