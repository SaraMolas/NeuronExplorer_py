# Script to plot different behavioural variables in sessions below versus above stability threshold and perform statistics to test significance 

# 1. Import packages

from tkinter import filedialog
import numpy as np 
from sklearn import svm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ranksums

# 2. Specify directory and load data

directory = filedialog.askdirectory()
print("Selected directory:", directory)
numberLicks = np.load (directory + "numberLicks.npy")
numberTrials = np.load (directory + "numberTrials.npy")
pausingTime = np.load (directory + "pausingTime.npy")
PCstab = np.load (directory + "PCstability.npy") 
preRewLicks = np.load (directory + "preRewardLicks.npy")
preRewSlope = np.load (directory + "preRewardSlope.npy")
licksC = np.load(directory + "roomCLicks.npy")
speedC = np.load(directory + "roomCspeed.npy")
speed = np.load(directory + "runningSpeed.npy")

# 3. Plot a violinplot for each behavioural variable comparing sessions with an average stability score below and above the threshold 0.5

# Prepare data for plotting
binaryStability = (PCstab >= 0.5).astype(int)
dataset = pd.DataFrame({'Number of trials': numberTrials.flatten(), 'Number of licks': numberLicks.flatten(), 'Pausing time': pausingTime.flatten(), 'Speed': speed.flatten(),
                        'Pre-reward licks': preRewLicks.flatten(), 'Pre-reward slope': preRewSlope.flatten(), 'Licks in room C': licksC.flatten(), 'Speed in room C': speedC.flatten(),
                        'Neural stability': binaryStability.flatten()})


# Plot number trials vs PC stability
sns.violinplot(data=dataset, x='Neural stability', y='Number of trials') # create violinplot
# Add labels, title and show
plt.ylabel('Number of trials')
plt.xlabel('Neural stability')
plt.title('Violin Plot of number of trials in sessions with low vs high neural stability')
plt.show()

# Plot number of licks vs PC stability
sns.violinplot(data=dataset, x='Neural stability', y='Number of licks') # create violinplot
# Add labels, title and show
plt.ylabel('Number of licks')
plt.xlabel('Neural stability')
plt.title('Violin Plot of number of licks in sessions with low vs high neural stability')
plt.show()

# Plot pausing time vs PC stability
sns.violinplot(data=dataset, x='Neural stability', y='Pausing time') # create violinplot
# Add labels, title and show
plt.ylabel('Pausing time')
plt.xlabel('Neural stability')
plt.title('Violin Plot of pausing time in sessions with low vs high neural stability')
plt.show()

# Plot speed vs PC stability
sns.violinplot(data=dataset, x='Neural stability', y='Speed') # create violinplot
# Add labels, title and show
plt.ylabel('Speed')
plt.xlabel('Neural stability')
plt.title('Violin Plot of speed in sessions with low vs high neural stability')
plt.show()

# Plot Pre-reward licks vs PC stability
sns.violinplot(data=dataset, x='Neural stability', y='Pre-reward licks') # create violinplot
# Add labels, title and show
plt.ylabel('Pre-reward licks')
plt.xlabel('Neural stability')
plt.title('Violin Plot of Pre-reward licks in sessions with low vs high neural stability')
plt.show()

# Plot Pre-reward slope vs PC stability
sns.violinplot(data=dataset, x='Neural stability', y='Pre-reward slope') # create violinplot
# Add labels, title and show
plt.ylabel('Pre-reward slope')
plt.xlabel('Neural stability')
plt.title('Violin Plot of Pre-reward slope in sessions with low vs high neural stability')
plt.show()

# Plot licks in room C vs PC stability
sns.violinplot(data=dataset, x='Neural stability', y='Licks in room C') # create violinplot
# Add labels, title and show
plt.ylabel('Licks in room C')
plt.xlabel('Neural stability')
plt.title('Violin Plot of Licks in room C in sessions with low vs high neural stability')
plt.show()

# Plot speed in room C vs PC stability
sns.violinplot(data=dataset, x='Neural stability', y='Speed in room C') # create violinplot
# Add labels, title and show
plt.ylabel('Speed in room C')
plt.xlabel('Neural stability')
plt.title('Violin Plot of Speed in room C in sessions with low vs high neural stability')
plt.show()

# 4. Perform a non-parametric statistical test (given the low number of data points in our dataset), to compare each behavioural indicator in sessions with low versus high neural stability

# Perform Wilcoxon rank sum test comparing number of trials in low vs high stability sessions
statistic, p_value = ranksums(dataset.loc[dataset['Neural stability'] == 0, 'Number of trials'], dataset.loc[dataset['Neural stability'] == 1, 'Number of trials'])
print("Wilcoxon Rank Sum Test for Number of trials, statistic: ", statistic, " p-value: ", p_value)

# Perform Wilcoxon rank sum test comparing number of licks in low vs high stability sessions
statistic, p_value = ranksums(dataset.loc[dataset['Neural stability'] == 0, 'Number of licks'], dataset.loc[dataset['Neural stability'] == 1, 'Number of licks'])
print("Wilcoxon Rank Sum Test for Number of licks, statistic: ", statistic, " p-value: ", p_value)

# Perform Wilcoxon rank sum test comparing pausing time in low vs high stability sessions
statistic, p_value = ranksums(dataset.loc[dataset['Neural stability'] == 0, 'Pausing time'], dataset.loc[dataset['Neural stability'] == 1, 'Pausing time'])
print("Wilcoxon Rank Sum Test for Pausing time, statistic: ", statistic, " p-value: ", p_value)

# Perform Wilcoxon rank sum test comparing Speed in low vs high stability sessions
statistic, p_value = ranksums(dataset.loc[dataset['Neural stability'] == 0, 'Speed'], dataset.loc[dataset['Neural stability'] == 1, 'Speed'])
print("Wilcoxon Rank Sum Test for Speed, statistic: ", statistic, " p-value: ", p_value)

# Perform Wilcoxon rank sum test comparing Pre-reward licks in low vs high stability sessions
statistic, p_value = ranksums(dataset.loc[dataset['Neural stability'] == 0, 'Pre-reward licks'], dataset.loc[dataset['Neural stability'] == 1, 'Pre-reward licks'])
print("Wilcoxon Rank Sum Test for Pre-reward licks, statistic: ", statistic, " p-value: ", p_value)

# Perform Wilcoxon rank sum test comparing Pre-reward slope in low vs high stability sessions
statistic, p_value = ranksums(dataset.loc[dataset['Neural stability'] == 0, 'Pre-reward slope'], dataset.loc[dataset['Neural stability'] == 1, 'Pre-reward slope'])
print("Wilcoxon Rank Sum Test for Pre-reward slope, statistic: ", statistic, " p-value: ", p_value)

# Perform Wilcoxon rank sum test comparing Licks in room C in low vs high stability sessions
statistic, p_value = ranksums(dataset.loc[dataset['Neural stability'] == 0, 'Licks in room C'], dataset.loc[dataset['Neural stability'] == 1, 'Licks in room C'])
print("Wilcoxon Rank Sum Test for Licks in room C, statistic: ", statistic, " p-value: ", p_value)

# Perform Wilcoxon rank sum test comparing Speed in room C in low vs high stability sessions
statistic, p_value = ranksums(dataset.loc[dataset['Neural stability'] == 0, 'Speed in room C'], dataset.loc[dataset['Neural stability'] == 1, 'Speed in room C'])
print("Wilcoxon Rank Sum Test for Speed in room C, statistic: ", statistic, " p-value: ", p_value)