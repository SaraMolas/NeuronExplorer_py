# NeuronExplorer_py
Repository to investigate electrophysiological neural data recorded by invasive brain implants in mice using Python. 

## Description 
This GitHub repository contains a collection of scripts I developed during my PhD in Neuroscience. These scripts were instrumental in investigating patterns of neural activity, derived from electrophysiological recordings of hippocampal place cells in mice. Specifically, the repository delves into three key facets of neural activity:

1. **NeuralAssemblies**: identifies neural assemblies (neurons that are active together) and then determines the ratio of a specific neural type (in my case 'disambiguating cells') present in the same cell assembly. This script is based on the method developed by Lopes-dos-Santos et al (2014) and Van de Ven et al (2016). First of all, a spike matrix is created with number of rows = number of neurons and number of columns = number of time bins. Each element of that matrix represents the number of spikes per time bin for each cell. Thus it is an integer >= 0. Next the covariance matrix is obtained from the spike matrix and the number of **eigenvalues** is calculated. The eigenvalues of the correlation matrix of a normal random matrix M with statistically independent rows follow a probability function described by the Marcenko-Pastur distribution. Lopes-dos-Santos et al. (2011) demonstrated that the upper bound of the Marcenko–Pastur distribution represents a statistical threshold for assembly activity, i.e., the number of cell assemblies in the network is estimated by the number of eigenvalues above this threshold. Any eigenvalue above that upper bound is considered a significant eigenvalue. The number of significant eigenvalues is equal to the number of neural assemblies. Then, a **fast ICA** (a dimensionality-reduction technique) with number of components = number of neural assemblies is performed on the spike matrix computed earlier. The output of the fast ICA is the weight vector of each neural assembly. This weight vector will assign a weight to all recorded neurons. The principal neurons of a neural assembly will be the neurons with a weight that is 2 standard deviations above the mean weight for that neural assembly. Finally, the ratio of the cell type that we are interested in (in this case, 'disambiguating cells') present in each neural assembly out of all principal neurons in the neural assembly will be computed. As a control, a number of place cells, equivalent to the number of disambiguating cells in that session, will be randomly selected from the list of total place cells recorded in that session, and a ratio of these randomly-selected place cells out of all principal neurons in each neural assembly will also be calculated. This step allows us to determine if the co-activation of disambiguating cells within a neural assembly is higher, lower or equal than expected by chance, by performing a **Wilcoxon rank-sum test** between the two mean ratios across sessions. 
   
4. **NeuralStability**: computes how stable the activity of each neuron was and investigates the relationship between neural stability and mice behaviour in the task. takes as input the ratemaps of hippocampal place cells in odd and even trials, as well as a few behavioural variables of each recording session. Then the algorithm computes the stability score of each place cell, which is the Pearson's correlation of the ratemap between odd and even trials. In other words, it measures how reliable the activity of that cell is across trials. Next, it plots a histogram illustrating the stability score of all recorded place cells. Following this, a **Support Vector Machine (SVM)** is trained to predict the average stability score of each session based on the behavioural indicators of a given session. Several average stability score thresholds and SVM hyperparameters are tested to reach peak performance. Finally, after finding the optimal stability threshold, in order to confirm the hypothesis that 'Hippocampal cell stability depends on mice engagement in the task', a series of **Wilcoxon rank-sum test** are performed to assess if there is a significant difference between behaviour and neural stability across sessions. This hypothesis is based on the paper by Pettit et al, 2022. 

5. **RoomPreferingNeurons**: in a task in which a mouse is running across multiple rooms (in my experiment 4 rooms), this script identifies the place cells that are active in the same location in each room, but that have a greater firing rate (are more active) in only one of the rooms ('the prefered room'). This is based on the 'lap-prefering cells' identified by Sun et al (2020) and their analytical approach. First of all, the **Pearson's correlation** between binned speed and binned firing rate within the place fields of all rooms is computed for each candidate neuron. If the correlation is significant (p-value < 0.05), that neuron is discarded, since the difference in activity(firing rate) across rooms might be due to differences in speed (behaviour) rather than an actual encoding of the room number by the neural firing rate. Following this, a **Monte Carlo Simulation** is performed to determine if the activity patterns of the candidate cells could have been obtained by chance. In particular, 1000 random shuffles are performed of the spatial bins of each neuron, this is done in such a way that the bin in position 1 in room X in trial A will be randomly shuffled to the bin in position 1 in room Y in trial B. This approach results in a distribution of firing rates for each bin in each room across trials per session and a mean ratemap over all trials is computed for that shuffle. Then the average peak firing rate of that cell is compared to the distribution of firing rates in that spatial bin. If the firing rate is in the top 5th percentile of the distribution, then it is considered that the firing rate is significantly greater than expected by chance and that cell is classified as a 'room-prefering neuron'. 

## Analytical methods used: 

- Fast Independent Component Analysis (fast ICA)
- Non-parametric statistics: Wilcoxon rank-sum test
- Support Vector Machine classifier
- Pearson's correlation
- Monte Carlo Simulation

## Python packages required: 

- For data handling and processing: **Numpy, Pandas, Math**
- For Statistics and Machine Learning: **Scikit-Learn (sklearn), Scipy.stats**
- For plotting: **Matplotlib.pyplot, Seaborn**
- For loading data/working with filenames: **os, tkinter**
  
## Glossary: 

- Place cell: neuron in the hippocampus that fires in a specific location in space.
- Place field: location in which a place cell is active. 
- Ratemap: average activity (aka firing rate in spikes per second) of a neuron in each spatial bin of an environment explored by an animal. If the environment is a linear track, the ratemap will be a 1-D vector. If the environment is an open arena, the ratemap will be a 2-D matrix.
- Hippocampus: brain structure involved in memory and spatial navigation. 
  
## References

- Lopes-dos-Santos V et al (2014). Detecting cell assemblies in large neuronal populations. J Neurosci Methods. 220(2):149-66. doi: 10.1016/j.jneumeth.2013.04.010. 
- Pettit NL et al (2022). Hippocampal place codes are gated by behavioral engagement. Nat Neurosci. 25(5):561-566. doi: 10.1038/s41593-022-01050-4.
- Sun C et al (2020). Hippocampal neurons represent events as transferable units of experience. Nat Neurosci. 23(5):651-663. doi: 10.1038/s41593-020-0614-x.
- Van de Ven GM et al (2016). Hippocampal Offline Reactivation Consolidates Recently Formed Cell Assembly Patterns during Sharp Wave-Ripples. Neuron. 92(5):968-974. doi: 10.1016/j.neuron.2016.10.020.

