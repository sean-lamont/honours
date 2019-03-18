
# Generalised Aitchinson Embeddings
Main contribution of this paper is the approach of learning the embedding of Compositional Data to Euclidean Space. This is done using a generalised version of the log embedding, which has as parameters a pseudo count to weight the importance of each component, and a linear transformation P. 

### Learning approach:
The embedding parameters are learnt using several metric learning approaches derived in the paper. These are the Alternating Optimisation (AO) and Projected subgradient descent with Nesterov acceleration (a momentum based approach, which can be improved by adaptive restart, which resets the momentum factor if it is decreasing performance). There is also a low-rank method which is more computationally efficient at the cost of convexity in the optimisation problem. 

Paper discusses related work which turn out to be specific cases of the general formualtion given here. Also Hellinger's Embedding, (i.e. Square root transformation) and others are mentioned. Maybe we could try different embeddings and see how they compare? 

### Histogram: 
-  Defined in this paper as the "normalised representation of bags of features (see below)"
-  I see this as being just the normalised representation of values i.e. ratios or percentages
-  Mathematically, the probability simplex (i.e. non negative components which sum to one)
-  Identical to the definition of Compositional Data? 
-  From this paper it is hard to tell if a histogram is an element of the simplex, or if they mean it as the unnormalised vector of values

### Bag of Features (BoF):
-  (https://www.researchgate.net/publication/225235567_Visual_Pattern_Analysis_in_Histopathology_Images_Using_Bag_of_Features)
-  Extract features from an image, then construct a codebook of a fixed size of features. 
-  BoF is then a histogram of the frequencies of each codeword in an image
-  BoF used in Le paper seems to denote frequency counts for any input data (rather than just images). I'm not sure what definition to take here. 

### Experiments
- Scene, image and text classification (also MNIST): All supervised learning problems
- SIFT and SURF are techniques used for feature extraction from the images and scenes, which are then represented as BoF as described above. (Interesting way to do classification: converting it into a lower dimentional histogram representation)
- For text, Bag of Words (BoW) feature representation is used, then reduced using Latent Dirichlet Modelling (LDA) (might be something to investigate?)
