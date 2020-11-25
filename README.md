
# CMSC 603 Assignment 1 MPI

John Naylor

### abstract

In an effort to understand the relationship between parallel algorithms and the speed up created by modifying the number of processes and cores available to these algorithm, K-nearest neighbors, which has a $O(n^2)$ runtime complexity, was modified to be parallel using a producer/consumer model. The modified KNN was executed with varying numbers of available processes and the duration was compared against the original unmodified KNN. It was discovered that the of all the trial runs, the maximum speed up achieved was a $14.85$ times speed up. 
