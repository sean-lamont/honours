# Week 6 Pre Meeting Notes
## Last Week:
 - Ran through regression experiments with Aitchinson data in more detail, only one dataset seemed to have decent results (others were too small, every method was almost equally poor). Using Christian's experimental cross val setup, there were clearly better results for CoDA-PCA than CLR (and naive regression) in the one example which had (somewhat) more data
 - Expanded on the proof of CoDA-PCA loss making it understandable (took me longer than I thought, since some assumptions/notation weren't made explicit in the paper)
 - Tried to understand the implementation of CoDA-PCA. The parametric TensorFlow Implementation was quite complicated (and with no experience there, was hard to understand), but the non parametric way with L-BFGS makes sense and follows easily from the paper
 
## This Week:
- We said we should have an idea of the topic the project is on. At this point, I'm leaning in the direction of linear regression, given that simple CoDA-PCA preprocessing seems to do well on the toy data (only one decent example here so could be misrepresentative). 
- - As discussed with Christian, one idea is to try set up an end to end optimisation problem (i.e. jointly optimise the regression loss and dimensionality reduction at the same time). Had a few ideas on this, like KKT/Convex constraints, some kind of deep learning inspired stacking of optimisation layers,...
- If we are in agreement, then will need to comb the regression for comp. data literature more thoroughly. Some initital topics include (in addition to hypershpere paper): 
-  - partial least squares http://cedric.cnam.fr/~saporta/compositional%20data%20analysis-1.pdf and https://link.springer.com/chapter/10.1007%2F978-3-540-32827-8_18
-  - Dirichlet regression https://cran.r-project.org/web/packages/DirichletReg/vignettes/DirichletReg-vig.pdf
-  - https://www.sciencedirect.com/science/article/pii/S0925231213005808