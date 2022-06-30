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
Credibility of Scientific research is based upon peer review and use to scientific community. You are expected to build a foundation of literature around new ideas. A Survey by Baker in 2016 asked authors publishing in the esteemed Nature magazin about reproducibility.
As you can see on the left:They found When replicating their own work, only a small majority of teams were not unsuccesful and generally arrive at the same conclusions. When working with other teams research, this picture is different. Many attempts already fail at reproducing the experiment or code, and then those results differ in a majority of cases.  
-> In more traditional fields, there is a pressure to publish in esteemed journals that is often required for grants or access to labs. Therefore, in some cases, 
 - authors act in bad faith and use methods like p hacking (collect data until hypothesis is true)
 - changing their hypothesis to be sucessful, to generate interesting results as this improves their chances to get published in a journal (MONYA BAKER NATURE article)
 - Other reasons include missing information about methodology or simply missing the means to access raw data can be seen on the right

Psychology also suffers from this issue and enact a possible solution:Pre-register studies.
- Before you start collecting data, fix hypothesis, number of samples, methodology, etc and get this peer reviewed (catch early mistakes)
- Some extra work but: If followed and everything is made accessible:
- Guarantee to be published in a magazine that is part of the Open Science Foundation.  

What does this have to do with ML? Intuitively, one would think that software is easier to work with than behavioural or chemical experiments. Isn't is just downloading code and pressing "RUN"? Assuming most of you 
tried to reimplement something you found published, are you agreeing with this statement? 
What we experience is sadly a wide spread problem in ML and NLP: Mieskes et al ACM community survey (2019) found that:
- In just 40% of cases teams other than the original team reached the same conclusion or figures. But in 30% of cases, they were able to get the system running but failed to obtain the similiar results. In the remaining cases, they did not manage to implement the system.
- Why is that?
- Public code is available on a Version Control System in just half the cases and in over a quarter of cases they are only available on the authors or lab page. Over 14% reported not being able to access the code in any way. 
- Public data is available on a VCS for 27% of cases and for 38% of cases on authors or lab pages. **11%** reported being unable to find data used in a paper.
- Issue? Lab and author pages may restructure, be taken down, etc
- For parameters, **40%** are found just reported in the paper while around **14%** of respondants were unable to find parameters (cmd line flags)
- In many of the remaining cases, code, data or parameters were obtained via personal communication via the original author - however, personal correspondence only yielded helpful answers in **30%** of contact attempts - others were ignored, not helpful or the 
responsible author left the lab and no process for contact was established.

Something else:Computers are difficult
- Python and Package Versions management via Anaconda or pyEnv is often the first issue but can resolved many times, although often resulting in a different use of versions
- Many packages like PyTorch use a source of randomness to speed up many operations. This can lead to a difference that is significance, but setting all seeds is often not enough to avoid this. However, at least for pytorch, there are deterministic versions of some operations leading to more stable results at the cost of performance
-  Implementations of algorithms can change the outcome of a model or an evaluation. Belz 2020 discusses that for a given NLP system, four reproductions trying to replicate the paper as close as possible returned results between 68-72% - a difference which may be considered a massive improvement. One unknown variable? Different Implementations of the weighted F1 evaluation. 
-  Even the same operations on different hardware can change outcome, especially on GPU (NVIDIA)
____
Belz (2019) discusses, that for multiple reproduction of Vajjala and Rama's Essay Grading System,
in which four different authors tried to mimic the original conditions as close as possible, scores of **68-72%** were reached - and a difference in 4 percent
is considered a massive improvement for an ML system. Here, the environment and score implementation, while the same algorithm, were not identical.   
____
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
