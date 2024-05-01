# Script to find room-prefering neurons 
# In a task in which mice run across 4 rooms, these neurons fire in the same position within each room, but the firing rate (spikes per second) is higher in only one of the rooms. Similar 
# to the lap-prefering cells identified by Sun et al (2020)

# 1. Import parameters
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats.stats import pearsonr
    
# 2. Specify directories to be used, load table with info about recording sessions and delete pre-existing variables with the same name
directory = filedialog.askdirectory()
print('Extract data from this directory: ', directory)
directory_fig = filedialog.askdirectory()
print('Save figures to this directory: ', directory_fig)
concatTable = pd.read_csv ("D:\Sara\ConcatenatedFiles\DataAnalysis2IR_finalVersionNew15.csv")
if 'countCells' in locals():
    del countCells 
    del ratemaps4PFs 

# 3. Iterate over sessions and skip sessions that we are not interested in
for day in range(0, concatTable.shape[0]): 

    if concatTable['sessions'][day]== 2: # if it's a sleep session, skip
        continue
    
    if concatTable['PCstability'][day] < 0.5: # if it's an unstable session, skip
        continue

    if concatTable['useIncompleteSessions'][day] == 0: # if it's a data-damaged session, skip
        continue

    # 4. LOAD DATA OF THAT SESSION ----------------------------------------------------------------------------------------------------------------------
    # extract some parameters from the info table to find the correct file 
    filepath = concatTable['output_path'][day]  
    mouse = concatTable['mice'][day]
    date = concatTable['dates'][day]
    session = concatTable['sessions'][day]

    for filename in os.listdir(directory + filepath + "/Analysis/forPython"):
        print('working on file: ', filename)

        if filename.startswith("ratemapsTrials_session_" + str(session) + "_cell_"): # if that session contains a candidate cell, a file starting with these letters will be present
            print("working on data from: " + str(filepath))

            # Load the numpy array containing the ratemap matrix of that cell (each value is spikes per second) and the speed matrix (each element is in cm/s)
            ratemapCell = np.load(directory + filepath + "/Analysis/forPython/" + filename) # dimension 1 = trials, dimension 2 = spatial bins
            speedBins = np.load(directory + filepath + "/Analysis/forPython/speedPerTrial_session_" + str(session) + '.npy') # dimension 1 = trials, dimension 2 = spatial bins

            # Obtain the cell ID of the ratemap that was just loaded and load the field position file (indicating where that neuron had the place field)
            numbers = []
            for char in filename: 
                if char.isdigit():
                    numbers.append(int(char))
            str_number = [str(num) for num in numbers]
            cell = ''.join(str_number[1:])
            fieldPos = np.load(directory + filepath + "/Analysis/forPython/posPF_session_" + str(session) + '_cell_' + cell + '.npy') # binary vector, dimension 1 = spatial bins 

            # Compute the mean ratemap across trials of each cell and store it together with the mean ratemaps of all cells in one 2D numpy array
            if 'ratemaps4PFs' in locals(): # if numpy array exists, append ratemap
                    ratemaps4PFs = np.append(ratemaps4PFs, np.reshape(np.nanmean(ratemapCell, axis=0)/np.max(np.nanmean(ratemapCell, axis=0)), [1,200]), axis=0)  # rows = cell number, columns = spatial bins    
            else: # if numpy array does not exist, create
                    ratemaps4PFs = np.reshape(np.nanmean(ratemapCell, axis=0)/np.max(np.nanmean(ratemapCell, axis=0)), [1,200])

            # Also store the cell number in a separate vector
            if 'countCells' in locals():
                countCells = np.append(countCells,  countCells[-1] + 1)
            else:
                countCells = np.array([1])
            
            # 5. COMPUTE AND PLOT CORRELATION BETWEEN SPEED AND FIRING RATE -----------------------------------------------------------------
            # Compute Pearson's correlation between binned speed and binned firing rate within the place fields of the candidate cell
            # first do some data preprocessing 
            flatRatemap = ratemapCell[:,fieldPos.flatten()].flatten() # extract the firing rates within the place field
            flatSpeed = speedBins[:,fieldPos.flatten()].flatten() # extract the speed within the place field
            nanInspeed = np.array(np.where(~np.isnan(flatSpeed)))
            nanInRatemap = np.array(np.where(~np.isnan(flatRatemap)))
            valuesToMask = np.intersect1d(nanInspeed, nanInRatemap)
            newSpeed = flatSpeed[valuesToMask]
            newRatemap = flatRatemap[valuesToMask]
            
            # Compute Pearson's correlation, extracting the correlation and the p-value
            r, p = pearsonr(newSpeed[newRatemap>0],newRatemap[newRatemap>0])

            # Create, display and save scatter plot of the binned speed versus binned firing rate within the place fields of the candidate cell
            plt.rcParams.update({'font.size': 30})
            plt.figure(figsize=(20*cm, 20*cm))
            plt.scatter(newSpeed[newRatemap>0], newRatemap[newRatemap>0])
            plt.xlabel('Binned speed (cm/s)')
            plt.ylabel('Binned firing rate (spikes/s)')
            plt.xlim(0, 100)        
            plt.title('r =' + str(np.round(r, 2)) + ', p = ' + str(np.round(p, 3))) # specify the correlation and p-value in the title 
            plt.savefig((directory_fig + 'scatterSpeedFiringRate_cellnum' + str(countCells[-1]) +'_m' + str(mouse) + '_' + str(date) + '_s' + str(session) + '_cell_' + cell + '.png'), dpi=300, bbox_inches='tight')
            plt.show()

            # Store correlation values for later 
            if 'corrSpeedRate' in locals(): # if variable already exists, append value to it
                corrSpeedRate = np.append(corrSpeedRate, r) 
                pSpeedRate = np.append(pSpeedRate, p) 
            else: # if variable does not exist, create it
                corrSpeedRate = np.array([r])
                pSpeedRate = np.array([p]) 
        
            # 6. MONTE CARLO SIMULATION -----------------------------------------------------------------------------------------------------------
            # Compute 1000 shuffles: shuffle activity of cells across trials and rooms (but maintaining spatial position within rooms).Then substract mean predicted activity from the shuffles 

            # Prepare data    
            shuffles = 1000 # specify number of shuffles
            stackedRatemap = np.concatenate((ratemapCell[:,0:50], ratemapCell[:,50:100], ratemapCell[:,100:150], ratemapCell[:,150:200]), axis = 0) # extract the spatial bins 
                # from the track corresponding to rooms, stacking the rooms in the row axis. So that now number of rows = number of trials * number of rooms, and number of columns is the 
                # number of spatial bins within a room
            shuffledRatemaps = np.empty((stackedRatemap.shape[0], stackedRatemap.shape[1], shuffles)) # initialize variable

            # Run shuffles, permutating spatial bins across rows
            for i in range(shuffles):
                for column in range(0, stackedRatemap.shape[1]):
                    shuffledRatemaps[:,column,i] = np.random.permutation(stackedRatemap[:,column])

            # Re-structure ratemap into number of columns = number of rooms * number of spatial bins within room
            shuffledRatemap = np.concatenate((shuffledRatemaps[0:ratemapCell.shape[0], :,:], shuffledRatemaps[ratemapCell.shape[0]:ratemapCell.shape[0]*2, :,:],
                                              shuffledRatemaps[ratemapCell.shape[0]*2:ratemapCell.shape[0]*3, :,:], 
                                                              shuffledRatemaps[ratemapCell.shape[0]*3:, :,:]), axis = 1)    

            # Compute mean ratemap for each cell and mean ratemap for each shuffle of that ratemap
            meanRatemap = np.nanmean(ratemapCell, axis=0)
            meanShuffledRatemap = np.empty((shuffledRatemap.shape[0], shuffledRatemap.shape[1], shuffledRatemap.shape[2]))
            for shuf in range(0, shuffledRatemap.shape[2]):
                meanShuffledRatemap = np.nanmean(shuffledRatemap[:,:,shuf], axis=0)

            # 7. CHECK IF THE CANDIDATE CELLS MEETS THE CRITERIA TO BE CLASSIFIED AS A ROOM-PREFERING CELL ------------------------------------------------------------
            # See if the cells peak (in overall ratemap) is in the top 5th percentile of shuffled activity and the correlation between speed and firing rate is not significant.
                # If so, save significant cell ratemap 
            peakInd = np.argmax(meanRatemap, axis=0)
            peakVal = meanRatemap[peakInd]
            peakDistribution = meanShuffledRatemap[0, peakInd, :]
            if peakVal >= np.percentile(peakDistribution, 95) and  pSpeedRate[-1] > 0.05:
                isCellSignificant = 1
                np.save(directory + filepath + "/Analysis/forPython/" + "meanRatemap_session_" + str(session) + "_cell_" + cell + "_significant.npy", meanRatemap) # dimension 1 = trials, dimension 2 = spatial bins
            else:
                isCellSignificant = 0
                            
            # Store results in a matrix
            if isCellSignificant == 1:
                normRatemap = meanRatemap/np.max(meanRatemap) # normalize ratemap for visualization purposes later
                if 'significantRatemaps' in locals():
                    significantRatemaps = np.append(significantRatemaps, np.reshape(normRatemap, (1,200)), axis=0)       
                    significantCells = np.append(significantCells, countCells[-1])
                else:
                    significantRatemaps = np.reshape(normRatemap, (1,200))
                    significantCells = np.array([countCells[-1]])

# Save results 
np.save(directory + 'significantRatemaps.npy', significantRatemaps)    
np.save(directory + 'significantCells.npy', significantCells)              

# 8. PLOT THE RATEMAPS OF ALL CANDIDATE CELLS AND THE RATEMAPS OF ALL ROOM-PREFERING CELLS

# 8.1. Plot ratemaps of candidate cells (neurons with a place field in each of the 4 rooms in our experiment)
ratemapPlot = np.repeat(ratemaps4PFs, 10, axis=0)
plt.rcParams.update({'font.size': 30})
plt.figure(figsize=(30*cm, 20*cm))
c = plt.imshow(ratemapPlot, vmin=0, vmax=1)
plt.colorbar(c)
plt.xlim(0, 200)    
ax = plt.gca()
ax.set_xticks(np.arange(0, 201, 50))
ax.set_yticks(np.arange(0, 150, 20))
ax.set_yticklabels(np.arange(0, 15, 2))
plt.xlabel("Position (in bins)", fontsize = 36)
plt.ylabel("Cell number", fontsize = 36)
plt.title("Ratemaps of place cells with \n one place field per room", fontsize = 36)
plt.savefig(directory_fig + 'ratemaps_candidateNeurons.png', dpi=300, bbox_inches='tight')
plt.show()                    

# 8.2. Plot ratemaps of all cells that meet the 'room-prefering neuron' criteria
significantRatemapsPlot = np.repeat(significantRatemaps, 10, axis=0)  # repeat all rows 10 times for visualization purposes         
plt.figure(figsize=(20*cm, 20*cm))
c = plt.imshow(significantRatemapsPlot)
plt.colorbar(c, shrink = 0.4)
plt.xlim(0, 200)    
ax = plt.gca()
ax.set_xticks(np.arange(0, 201, 50))
ax.set_yticks(np.arange(10, 90, 10))
ax.set_yticklabels(np.arange(1, 9, 1))
plt.xlabel("Position (in bins)")
plt.ylabel("Cell number")
plt.title("Ratemaps of room-preferring cells")
plt.savefig(directory_fig + 'ratemaps_roomPreferingNeurons.png', dpi=300, bbox_inches='tight')
plt.show()
