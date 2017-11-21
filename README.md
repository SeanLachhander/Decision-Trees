# Decision-Trees
Implementation of the three measures (Information Gain, CART, Gini Index).
Datasets included: train.txt, and test.txt
Each row contains 11 values - the first 10 are attributes (a mix of numeric and categorical translated to numeric (ex: {T,F} = {0,1}), and the final being the true class of that observation. The load function separates class from data.

** Information Gain **
(1) Implemented the Information Gain function where D is the dataset, index is the index of an attribute and value is the split value such that the split is of the form Xi <= value. The function returns the value of the Information Gain.

** Gini Index **
(2) Implemented the G(D, index, value) function where D is a dataset, index is the index of an attribute and value is the split value such that the split is of the form Xi ≤ value. The function returns the value of the Gini index value.

** CART **
(3) Implemented the CART(D, index, value), where D is a dataset, index is the index of an attribute, and value is the split value such that the split is of the form Xi ≤ value. The function returns the value of the CART value.

** BestSplit(D, criterion) Function **
(4) Implemented the function bestsplit(D, criterion) which takes as an input a dataset D, a string
value from the set {“IG", “GINI00, “CART"} which specifies a measure of interest. This function returns the best possible split for measure criterion in the form of a tuple (i, value), where i is the attribute index and value is the split value. The function probes all possible values for each attribute and all attributes to form splits of the form Xi ≤ value.

** Loading Information **
(5) Loaded the training data “train.txt” provided by implementing the function load('filename'), which returns a dataset D, and finds the best possible split for each of the three criteria for the data.
