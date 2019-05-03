# Introduction
This repository contains a collection of Matlab codes that perform MCMC estimation of a number of dynamic correlation models, including three existing models: [DCC (Engle 2002)](http://pages.stern.nyu.edu/~rengle/dccfinal.pdf), [VC (Tse and Tsui 2002)](https://core.ac.uk/download/pdf/7354515.pdf), [cDCC (Aielli 2013)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1507743) and four versions of the [proposed new model](https://mpra.ub.uni-muenchen.de/84820/1/MPRA_paper_84820.pdf) by myself: GDC, top-integrated GDC (TIGDC), bottom-integrated GDC (BIGDC), and both-levels-integrated GDC (IGDC). 

The DCC and VC models are the benchmarks. They each find a structure to summarize information in asset returns dynamically. The summarized information then goes through a normalization to ensure the constraint of a correlation matrix: positive definite with unit diagonal elements. The cDCC model builds on the DCC model to solve an estimation problem (in practice correlation estimates from the DCC and cDCC models are often pretty close). The proposed GDC model is a generalization of the DCC and VC models and nests them as special cases. Details of the GDC model can be found in my working paper ["A Class of Generalized Dynamic Correlation Models"](https://mpra.ub.uni-muenchen.de/84820/1/MPRA_paper_84820.pdf).

# How to Use the Codes
Users only need to use the "main_*.m" files

I am aware of the R package "bayesDccGarch" by Fiorucci et al. that performs Bayesian estimation of the DCC model. Nevertheless I find its implementation of the parameter restrictions unsatisfying. Specifically let a and b be the parameters governing the dynamics of the correlation matrix in the DCC model. We need not only 0<a<1 and 0<b<1 but also 0<a+b<1. From what I see, the package "bayesDccGarch" only enforces the former two constraints for individual parameters but not for the sum of the two parameters. In my implementation, I explicitly enforces 0<a+b<1.  

Matlab is choosen to write the codes because recent versions of Matlab are pretty fast (in my own experiences, faster than R and Python with the added benefit of good help files). I am aware of the non-open-source problem of Matlab (Octave is a nice open-source alternative to run the codes but would be painfully slow) and may consider writing R/Python/Julia codes in the future as open-source alternatives.

Comments and questions are welcome!
