# NeuronExplorer_py
Repository to investigate electrophysiological neural data recorded by invasive brain implants in mice using Python. 

## Description 
This GitHub repository contains a collection of scripts I developed during my PhD in Neuroscience. These scripts were instrumental in investigating patterns of neural activity, derived from electrophysiological recordings of hippocampal place cells in mice. Specifically, the repository delves into three key facets of neural activity:

1. **NeuralStability**: this jupyter notebook computes how stable the activity of each neuron was and investigates the relationship between neural stability and mice behaviour in the task. takes as input the ratemaps of hippocampal place cells in odd and even trials, as well as a few behavioural variables of each recording session. Then the algorithm computes the stability score of each place cell, which is the Pearson's correlation of the ratemap between odd and even trials. In other words, it measures how reliable the activity of that cell is across trials. Next, it plots a histogram illustrating the stability score of all recorded place cells. Following this, a **Support Vector Machine (SVM)** is trained to predict the average stability score of each session based on the behavioural indicators of a given session. Several average stability score thresholds and SVM hyperparameters are tested to reach peak performance. Finally, after finding the optimal stability threshold, in order to confirm the hypothesis that 'Hippocampal cell stability depends on mice engagement in the task', a series of **Wilcoxon rank-sum test** are performed to assess if there is a significant difference between behaviour and neural stability across sessions. This hypothesis is based on the paper by Pettit et al, 2022. 

2. **NeuralAssemblies**:
   
3. **RoomPreferingNeurons**: 

## Glossary: 

- Ratemap:
- Place cell:
- Hippocampus:
  
## References

- Lopes-dos-Santos V et al (2014). Detecting cell assemblies in large neuronal populations. J Neurosci Methods. 220(2):149-66. doi: 10.1016/j.jneumeth.2013.04.010. 
- Pettit NL et al (2022). Hippocampal place codes are gated by behavioral engagement. Nat Neurosci. 25(5):561-566. doi: 10.1038/s41593-022-01050-4.
- Sun C et al (2020). Hippocampal neurons represent events as transferable units of experience. Nat Neurosci. 23(5):651-663. doi: 10.1038/s41593-020-0614-x.
- Van de Ven GM et al (2016). Hippocampal Offline Reactivation Consolidates Recently Formed Cell Assembly Patterns during Sharp Wave-Ripples. Neuron. 92(5):968-974. doi: 10.1016/j.neuron.2016.10.020.

