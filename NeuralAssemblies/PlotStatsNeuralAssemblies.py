# Script to plot the average ratio of disambiguating cells and randomly-selected place cells out of the principal neurons of all neural assemblies across all sessions and perform stats

# 0. Import packages
from tkinter import filedialog
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists

# 1. Select working directory and load table with info about all recording sessions
directory = filedialog.askdirectory()
print('Selected directory: ', directory)
concatTable = pd.read_csv ("D:\Sara\ConcatenatedFiles\DataAnalysis2IR_finalVersionNew15.csv")

# 2. Initialize some variables and iterate over sessions
meanDisSessions = []
meanPlaceSessions = []

for day in range(0, concatTable.shape[0]): 

    # In this case, we are only interested in sessions that have a specific type of cell ('disambiguating cells'), if that session has disambiguating cells, a npy file will be stored in the 
        # corresponding folder
    if exists(directory + concatTable['output_path'][day] + "/Analysis/forPython/ratioDisCells" + str(concatTable['sessions'][day]) + "v2.npy"):
        print("working on data from: ", concatTable['output_path'][day])
        
        # 3. LOAD DATA OF THAT SESSION ----------------------------------------------------------------------------------------------------------------------
        filepath = concatTable['output_path'][day]  
        print("working on data from: " + str(filepath))
        ratioDisCells = np.load(directory + filepath + "/Analysis/forPython/ratioDisCells" + str(concatTable['sessions'][day]) + "v2.npy") 
        ratioRandomPlaceCells = np.load(directory + filepath + "/Analysis/forPython/ratioRandomPlaceCells" + str(concatTable['sessions'][day]) + "v2.npy")

        # 4. STORE MEAN RATIOS ACROSS SESSIONS ----------------------------------------------------------------------------------------------------------------------
        meanDisSessions.append(np.mean(ratioDisCells))
        meanPlaceSessions.append(np.mean(ratioRandomPlaceCells))    

# 5. PLOT RATIOS ----------------------------------------------------------------------------------------------------------------------
# violinplot the ratio of disambiguating cells among the principal cells of each assembly
directory_fig = filedialog.askdirectory() # select where to save figure
print('Save violinplot in directory: ', directory_fig)

cm = 1/2.54
plt.figure(figsize =(30*cm, 25*cm))
plt.violinplot([meanDisSessions,  meanPlaceSessions],showmeans = True)
plt.xticks([1, 2], ['Disambiguating cells', 'Place cells'], fontsize = 28)
plt.ylabel('Ratio in cell assembly', fontsize = 36)
plt.title('Mean ratio of cells \n among main cells in each assembly', fontsize=40)
plt.ylim(-0.02,0.17)
plt.savefig(directory_fig + 'violinplot_meanRatioDisCells.png')
plt.show()

# 6. PERFORM STATS ---------------------------------------------------------------------------------------------------------------------
statistic, p_value  = stats.ranksums(meanDisSessions,  meanPlaceSessions)
print('results of ranksum test: ', statistic, ' and p-value: ', p_value )

# 7. SAVE RESULTS  ---------------------------------------------------------------------------------------------------------------------
np.save("D:/Sara/Figures_analysis/Thesis/subsetCells/characteristics/meanDisCells.npy", meanDisSessions)
np.save("D:/Sara/Figures_analysis/Thesis/subsetCells/characteristics/meanRandomPlaceCells.npy", meanPlaceSessions)
