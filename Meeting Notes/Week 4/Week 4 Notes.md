# Week 4 Notes

## Meeting Points:
- 0's important
- is the maximum always one?
- Subcompositions: Subset of the columns are a composition
- Be careful if raw counts vs normalised 
Moving forward:
- Text applications..
- Something like: Given a text review predict # of stars (ordinal regression) (supervised approach)
- Chain logistic regression with NeurIPS (find python code!). 
- I.e. Pre process with NeurIPS PCA to 64 Dim Vector and then do logistic regression
- Correspondence analysis
- Find datasets which are histograms
- Supervised Learning probably the way to go since easy to verify
- Ideally 32 datasets...
- Big Enough for classification, not too many classes
- Methods for finding datasets:
	- Papers
	- Google
	- MLData.org
	- OpenML.org

- Aitchinson Book datasets in appendix.

- Aim for experiments done quickly i.e. within 3 minutes. 

- Unsupervised Learning to get vector features for words

- Correspondence Analysis for ^ 

- All genomics paper are compositional paper: look at datasets from this

- Benchmarks:- No preprocessing, Pre with standard PCA, Pre with CoDA PCA

- Aitchinson Book good start

## Goals

- Datasets
- CoDA-PCA Implementation, then regression

## Outcomes

## Datasets

### 1
From "All Microbiome Data are Compositional" Paper (https://www.frontiersin.org/articles/10.3389/fmicb.2017.02224/), they have example datasets with analysis and explanations. These data are suited to unsupervised methods, so may not be of as much interest if we pursue e.g. Linear Regression. 

Could try applying CoDA-PCA (instead of just PCA after log transformation) in their pipeline and compare the results. 

Data and code @: https://github.com/ggloor/CoDaSeq

Complicated and large datasets, probably best to leave any work with these until later on The paper is quite high impact and recent (2017, >98% of articles in a 4+ impact factor journal) so it would be a great place to revisit.

### 2
Aitchinson had 40 small compositional datasets for a variety of different topics, with some variance in the structure of the datasets. Worked with Ragib to collate these into csv files for use in this project, and also to host online for future research endeavours. These are good toy datasets to test with, since they are quite varied and are all compositional. 

### CoDA-PCA Implementation:
Spent quite some time reading through the implementation. Some notes:

- There are 2 methods that are implemented for both CoDA-PCA and S-CoDA-PCA in the source. One is SGD using tensorflow, and the other is L-BFGS. SGD is parametric and so 
- The loss function includes the subtraction of a trace, which is implemented by a hadamard product (I assume? It took me a while to find this since I wasn't actually sure where the trace was being implemented)
- Applied CoDA-PCA to some of Aitchinson's data, then applied linear regression. This outperformed CLR-PCA and naive regression. 