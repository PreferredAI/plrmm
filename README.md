# PLRMM (Plackett-Luce Regression Mixture Model)

PLRMM provides a Python implementation of an algorithm for finding clusters within a population of rankers and learning their
ranking functions. We call such clusters preference groups.  Each ranker is represented by the set of rankings it 
produces over a set of some items. The items are defined by their features.

PLRMM is a probability model that specifies mixture over observed rankings. Each preference group is identified by its weight
vector, which are regression coefficients used to transform a point in the item feature space to its score.
The scores are used to induce the ranking over the set of items one wants to rank.  The basic intuition is that the higher the score is,
the higher the position of the item in the ranking should be.  To be precise the scores define the probability distribution 
(known as Plackett-Luce model) over all possible item permutations and each observed ranking is a sample of this distribution.

If one is interested in a learning to rank model (without the mixture modeling), PLR (Plackett-Luce Regression) is a learning to rank model defined within each preference group
of PLRMM and can be modelled as a special case when the expected number of clusters is `K = 1`. 

Keywords: Plackett-Luce Model, Learning to Rank, Probability Model, Expectation Maximization

### How to Cite

The detailed description of the algorithms as well as the experiments can be found in the following paper:

> Maksim Tkachenko and Hady W. Lauw, Plackett-Luce Regression Mixture Model for Heterogeneous Rankings, CIKM 2016.

If you are using this library in your research, please consider to cite the above-mentioned paper.

### Computer Survey Motivation 

One can imagine a scenario where in a survey, `M` people are asked to rank `N` computers by
their preferences or likelihood of buying the computer. In total we obtain `M` rankings. In general,
there might be more than one rankings produced by the same subject. Each computer is represented
with specification attributes including CPU frequency, RAM amount, price, etc.

Based on the subject's background, experience, and requests, the rankings may vary from one person to another.  A computer gamer 
might prefer a computer with a better video card, while an ordinary buyer might be satisfied with the average computer configuration.
This observation implies different rankings from the different categories of users. It might be useful to identify these groups 
for further analysis and to learn their decision making process. And that is exactly what PLRMM does!

# Installation

```
python setup.py install
```

The library is implemented for Python 3 and requires a number of additional packages to be installed:
* NumPy (==1.11.0)
* SciPy (==0.17.1)
* Cython (==0.24)

In parenthesis, we specify the versions of the packages that should work fine with the PLRMM.

# Usage

PLRMM is designed for the use as standalone library that can be manipulated via a number of Python scripts.
In case you want to modify or use the model in your code, please refer to the source files. 

### Training PLRM Model
```bash
python plrmm_train.py [-k <K>] <examples> <model>
```

`-k` specifies the number of preference groups (clusters) to be discovered in the dataset.  
`<examples>` is the path to the dataset.  
`<model>` specifies an output file for the trained PLRMM.

The script accepts the number of other optional parameters, such as hyperparameters, number of iterations, etc., for additional 
information run the script with `-h` or `--help` option.  

### Assigning Group to New Ranker
```bash
python plrmm_predz.py <model> <examples> <assignment>
```

`<model>` is the path to the trained PLRMM.  
`<examples>` is the path to the dataset file, which is used to infer group assignment for the new rankers given that the
trained model is already provided and it should not be modified.  
`<assignment>` specifies an output file where the assignment have to be stored. 

### Predicting Rankings
```bash
python plrmm_predy.py <model> <examples> <assignment> <prediction>
```

`<model>` is the path to the trained PLRMM.  
`<examples>` is the path to the dataset file, which is used to predict the ranking, assuming that the group assignments for the 
rankers are provided.  
`<assignment>` is the path to the group assignment file.  
`<prediction>` specifies an output file for the ranking prediction.

# Data Format

### Input `<examples>`

This section specifies the training/test input formats. 

The first line of the file contains two integers `N` and `M`. `N` is the number of items available for ranking. `M` is the
number of rankings.

The next `N` lines specify the items in terms of their features. Each line describes only one item. It contains pairs
of integers `<index>:<value>` separated by spaces. A pair `<index>:<value>` specifies a feature value: `index` 
is non-negative integer, index of the feature; `value` is a real number, its value. Each of the `N` items can be
referred in the ranking section by a its implicitly assigned index, the first appeared item has index `0`, 
the next `1`, and so on.

The next `M` lines specify the rankings. Each ranking is a set of item indices separated by spaces. The first index identifies item 
that occupies the first position in the ranking, the second refers to the item that occupies the second position and etc. 
Each line is for one ranking.

```
<N> <K>
<index1>:<value1> <index2>:<value2> ... 
.
.
.
<item_index1> <item_index2> ...
.
.
.
```

Example of three one-hot 'identity' items and two opposite rankings:

```
3 2
0:1
1:1
2:1
0 1 2
2 1 0
```

### Ranking Output `<prediction>`

The ranking prediction file (produced by `plrmm_predy.py`) contains only ranking section. It rearranges the item indices with 
respect to the order implied by the trained model.

### Preference Group Assignment Output `<assignment>`

The group assignment file (produced by `plrmm_predz.py`) is a MATLAB data file (`*.MAT`) with two fields:
* `'pz'` contains `M x K` matrix that specifies the posterior probability distribution over the group assignments;
* `'assignment'` contains `1 x M` vector that specifies the most probable preference group for each ranking.

NOTE: If one ranker produces several rankings in the dataset, it might be useful to assign all these rankings to the same 
preference group. This block assignment strategy is not implemented yet at the inference phase, but one may use the 
probability over possible group assignments (`'pz'` field of the `*.MAT` file) to implement, for example, majority vote.

### Model `<model>`

A trained model is stored in MATLAB data file (`*.MAT`). For the accurate up-to-date information of the stored 
fields please refer to the source files.

## Copyright

Copyright 2016 Singapore Management University (SMU). All Rights Reserved. 

Permission to use, copy, modify and distribute this software and 
its documentation for purposes of research, teaching and general
academic pursuits, without fee and without a signed licensing
agreement, is hereby granted, provided that the above copyright
statement, this paragraph and the following paragraph on disclaimer
appear in all copies, modifications, and distributions.  Contact
Singapore Management University, Intellectual Property Management
Office at iie@smu.edu.sg, for commercial licensing opportunities.

This software is provided by the copyright holder and creator "as is"
and any express or implied warranties, including, but not Limited to,
the implied warranties of merchantability and fitness for a particular 
purpose are disclaimed.  In no event shall SMU or the creator be 
liable for any direct, indirect, incidental, special, exemplary or 
consequential damages, however caused arising in any way out of the
use of this software.
