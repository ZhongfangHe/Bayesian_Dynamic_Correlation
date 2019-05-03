# Introduction
This repository contains a collection of Matlab codes that perform MCMC estimation of a number of dynamic correlation models, including three existing models: [DCC (Engle 2002)](http://pages.stern.nyu.edu/~rengle/dccfinal.pdf), [VC (Tse and Tsui 2002)](https://core.ac.uk/download/pdf/7354515.pdf), [cDCC (Aielli 2013)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1507743) and four versions of the [proposed new model](https://mpra.ub.uni-muenchen.de/84820/1/MPRA_paper_84820.pdf) by myself: GDC, top-integrated GDC (TIGDC), bottom-integrated GDC (BIGDC), and both-levels-integrated GDC (IGDC). 

The DCC and VC models are the benchmarks. They each find a structure to summarize information in asset returns dynamically. The summarized information then goes through a normalization to ensure the constraint of a correlation matrix: positive definite with unit diagonal elements. The cDCC model builds on the DCC model to solve an estimation problem (in practice correlation estimates from the DCC and cDCC models are often pretty close). The proposed GDC model is a generalization of the DCC and VC models and nests them as special cases. Details of the GDC model can be found in my working paper ["A Class of Generalized Dynamic Correlation Models"](https://mpra.ub.uni-muenchen.de/84820/1/MPRA_paper_84820.pdf).

Note that asset returns are assumed to follow normal distributions. Fat-tailed or other more complicated distributions are not covered by this version of the codes.

# How to Use the Codes
Users only need to modify the "main_*.m" files to estimate the various correlation models; the other files are auxiliary functions and should not be modified. 

Each "main_*.m" file has comments at the start that describes the model. As the same notations of the parameters are used in the main body of the code, users should read the model description to understand the meaning of the variables in the code and to use the model outputs properly. 

There is a double %% comment section "Inputs" that would require the users' inputs such as where to find the data spreadsheets (Excel), the hyper-parameters of the prior distributions, the tuning parameters to adjust the random-walk proposals in the Metropolis-Hastings steps, and where to write the output files (CSV). Each input variable has a in-line comment that describes the variable. Except for the "Inputs" section, other parts of the code should not be modified. The variable "draws" will contain the posterior draws of the model variables.

# Additional Comments
I am aware of the R package "bayesDccGarch" by Fiorucci et al. that performs Bayesian estimation of the DCC model. Nevertheless I find its implementation of the parameter restrictions unsatisfying. Specifically let a and b be the parameters governing the dynamics of the correlation matrix in the DCC model. We need not only 0<a<1 and 0<b<1 but also 0<a+b<1. From what I see, the package "bayesDccGarch" only enforces the former two constraints for individual parameters but not for the sum of the two parameters. In my implementation, I explicitly enforces 0<a+b<1.  

Matlab is choosen to write the codes because recent versions of Matlab are pretty fast (in my own experiences, faster than R and Python with the added benefit of good help files). I am aware of the non-open-source problem of Matlab (Octave is a nice open-source alternative to run the codes but would be painfully slow) and may consider writing R/Python/Julia codes in the future as open-source alternatives.

Comments and questions are welcome!
