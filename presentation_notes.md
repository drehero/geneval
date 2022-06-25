# Reproducibility

## What is reproducitility? What is the difference to repeatability?
Some definitions  
### ACM:
results have been reproduced if obtained in a different study by a different team using artifacts supplied in part by the original authors  
replicated if obtained in a different study by a different team using artifacts not supplied by the original authors.  
### Drummond (2009)  
what ML calls reproducibility is in fact replicability which is the ability to re-run an experiment in exactly the same way, whereas true
reproducibility is the ability to obtain the same result by different means. 
### Rougieret al. (2017)
[r]eproducing the result of a computation means running the same software on the same input data and obtaining the same results. 
Replicating a published result means writing and then running new software based on the description of a computational model or method provided in
the original publication”.

(asarXiv:2109.01211v1 [cs.CL] 2 Sep 2021)

Altogether quite unsatisfying definitions (we will come to a newly suggested one later) 

## Why is this important? 
-> Reproducability Crisis
Scientific research is based upon peer review:
However, in many fields, data is not openly available or exact methodolgy is unclear. When replicating their own work, most teams are succesful
and generally arrive at the same conclusions
When working with other teams research, this picture is different. Many attempts already fail at reproducing the experiment or code, and then those results
differ in a majority of cases.

-> In some cases, authors act in bad faith and use methods like p hawking (collect data until hypothesis is true) 
or changing their hypothesis to be sucessful, to generate interesting results as this improves 
their chances to get published in a journal (MONYA BAKER NATURE article)
Psychology recognised this issue and there is a push to pre-register studies, in which all parameters eg. number of data points, methods and hypothesis 
are fixed apriori. 
What does this have to do with NLP? Intuitively, one would think that software is easier to work with than behavioural experiments. Assuming most of you 
tried to reimplement something, are you agreeing with this statement? 
What you experienced is sadly a wide spread problem. Often finding the original code let alone the data is of issue. Additionally, even if there is a 
repository, many are not well maintained, and it is unclear which version was used. 
In Mieskes et al ACM community survey (2019), they found that in 60% of cases, code could be obtained on github or the authors webpage, over 14% reported 
not being able to find any public code. **11%** reported being unable to find data used in a paper. For parameters, **40%** are found just reported in the paper
while around **14%** of respondants were unable to find parameters. In many other cases, code, data or parameters were obtained via personal communication via 
the original author - however, personal correspondence only yielded helpful answers in **30%** of contact attempts - others were ignored, not helpful or the 
responsible author left the lab.  
An issue not yet discussed is the package environment and exact procedure used. Belz (2019) discusses, that for multiple reproduction of a NLP implementation,
in which four different authors tried to mimic the original conditions as close as possible, scores of **68-72%** were reached - and a difference in 4 percent
is considered a massive improvement for an ML system. Here, the environment and score implementation, while the same algorithm, were not identical.   

## What is being done? 
This issue has gained traction, including being a task in several conferences (ICML,ICRL, NEURIPS) and even the main focus of others (REPROLANG,ReproGen). 
There are several checklist like https://github.com/paperswithcode/releasing-research-code in which a template for providing code, parameters and data are
given. However, there have been attempts to streamline and alingn the defintion across science. One is given bei Belz (2019,2020) with their 
### Quantified Reproducibility Assessment:
Based on Metrology (Science of Science)
  
Reproducibility is a measure of precision of repetitions of the given measurement under a set of of conditions that differs in some condition each other  
Repeatability is a special form of reproducibility, in which all conditions are the same.  
Precision can be choosen, but coefficient of variance is a great choice, as it is unit independent.  

____
- 1.For a set of n measurements to be assessed, identify the shared object and measurand.  
- 2. Identify all conditions of measurement Ci for which information is available for all measurements, and specify values for each condition, including measurement method and procedure.
- 3. Gather the n measured quantity values v1; v2; :::vn.  
- 4. Compute precision for v1; v2; :::vn, giving reproducibility score R.  
- 5. Report resulting R score and associated confidence statistics, alongside the Ci.  
____
Example:
- Object: ML System  
- Measurand: BLUE SCORE
- Object Conditions: Code, Compile, training procedure 
- Measurement Conditions: Defintion, Implementation (code, **our library**)
- Measurement Procedure condition: Procedure, Test Set, Responsible Party

Show example

Quantifying Reproducibility in NLP and ML (Figure 3)

Given a set of measurements, one can compute the precision. Now, the scientific community needs to establish limits of precision that are acceptable to
fight this reproducibility crisis.













Mieskes, M., Fort, K., Névéol, A., Grouin, C., & Cohen, K. B. (2019, September). NLP Community Perspectives on Replicability. In Recent Advances in Natural Language Processing.
