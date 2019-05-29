# Week 11 Notes
## American Gut Microbiome

The main paper I found (linked below) could be interesting to compare results to,
with many of there results and plots coming from standard PCA and Principal Coordinate Analysis.


(for example, they compare food frequency questionares (FFQ) (I believe this would be compositional) to the microbiome results,

but I don't think the FFQ data is available)


https://msystems.asm.org/content/msys/3/3/e00031-18.full.pdf


There is indeed a large amount of data, and the AGP github (https://github.com/biocore/American-Gut)
seems to have well formatted (though older) data. The newest data is on an ftp server (ftp://ftp.microbio.me/../../AmericanGut/)
which isn't well documented however.

The main data format is .biom, which is standard for OTU's, with Python (biom) and R (biomformat) packages for processing these.

Looks like a promising dataset, lots of metadata (and so potential regression/classification targets), with large sample size and dimensionality.

## Code Progress

Solid progress on the code this week:
- Completed the base CoDA-PCA PyTorch implementation
- Replicated results of the Tensorflow version
- Have the base code for the end to end model. Ideally will test this over the next few weeks,
 and commence experiments if everything works. 
