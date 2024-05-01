# Script to detect the neural assemblies in each session and the ratio of 'disambiguating cells' present in each neural assembly across all sessions 

# 0. Import packages
from tkinter import filedialog
import numpy as np
from scipy import stats
import pandas as pd
import math
from sklearn.decomposition import FastICA
import random 
from os.path import exists

# 1. Select working directory and load table with info about all recording sessions
directory = filedialog.askdirectory()
print('Selected directory: ', directory)
concatTable = pd.read_csv ("D:\Sara\ConcatenatedFiles\DataAnalysis2IR_finalVersionNew15.csv")

# 2. Iterate over all sessions
for day in range(0, concatTable.shape[0]): 

    # In this case, we are only interested in sessions that have a specific type of cell ('disambiguating cells'), if that session has disambiguating cells, a npy file will be stored in the 
        # corresponding folder
    if exists(directory + concatTable['output_path'][day] + "/Analysis/forPython/disambiguatingCells_session" + str(concatTable['sessions'][day]) + ".npy"):

        # 3. LOAD DATA OF THAT SESSION ----------------------------------------------------------------------------------------------------------------------
        filepath = concatTable['output_path'][day]  
        print("working on data from: " + str(filepath))
        # load spike data
        spikeTimes = np.load(directory + filepath + "/spike_times.npy") # vector of electrophysiology timestamps of all spikes
        spikeClusters = np.load(directory + filepath + "/spike_clusters.npy") # vector of cell ID of all spikes
        # load behavioural data
        timesRun = np.load(directory + filepath + "/Analysis/forPython/time_run" + str(concatTable['sessions'][day]) + ".npy") # vector with behavioural timestamps
        posRun = np.load(directory + filepath + "/Analysis/forPython/positions_run" + str(concatTable['sessions'][day]) + ".npy") # vector with positions for each behav timestamp 
        speedRun = np.load(directory + filepath + "/Analysis/forPython/speed_run" + str(concatTable['sessions'][day]) + ".npy") # vector of speed for each behav timestamp
        trialRun = np.load(directory + filepath + "/Analysis/forPython/trials_run" + str(concatTable['sessions'][day]) + ".npy") # vector of trial number for each behav timestamp

        # since the electrophysiology timestamps do not correspond to the behavioural timestamps due to different recording systems, we need to adjust them
        if concatTable['truncatedSize'][day] > concatTable['originalSize'][day]:
            print("not truncated")
            timeSes = concatTable['originalSize'][day] - concatTable['StartVirmenZeroeds'][day]
        else:
            print("truncated")
            timeSes = concatTable['truncatedSize'][day] 
        if concatTable['sessions'][day] == 1:
            startTime = 0 # spikes have already been cut so only comprise recording periods and not inter-VR-rest periods
            endTime = (timeSes + 1)/30000 # convert time from openephys sample points to seconds
        elif concatTable['sessions'][day] == 3: 
            counter = 0
            timeSes = np.zeros([3,1])
            for idx in range(-2,1):
                print (idx)
                if concatTable['truncatedSize'][day + idx] > concatTable['originalSize'][day + idx]:
                    timeSes[counter] = concatTable['originalSize'][day + idx] - concatTable['StartVirmenZeroeds'][day+idx]
                else: 
                    timeSes[counter] = concatTable['truncatedSize'][day + idx] 
                counter = counter + 1
            startTime = (timeSes[0] + timeSes[1] + 1) / 30000 # convert time from openephys sample points to seconds
            endTime = (timeSes[0] + timeSes[1] + timeSes[2]) / 30000 # convert time from openephys sample points to seconds
        print("start of session: " + str(startTime) + " and end of session: " + str(endTime))

        # 4. GET SPIKE MATRIX ----------------------------------------------------------------------------------------------------------------------
        print('Computing cell assemblies')
        timestamps = np.arange(timesRun[0],math.ceil(timesRun[-1]), 0.025) # get timestamps in 25 ms bins
        numberCells = np.unique(spikeClusters) # get number of cells
        cellsTime = np.zeros((len(numberCells), len(timestamps))) # create matrix with zeros
        spikeMatrix = np.zeros((len(numberCells), len(timestamps))) # create matrix with zeros
        cells, timeBins = cellsTime.shape # extract number of cells and number of time bins

        # Fill matrix with spike count for each neuron per timebin
        # then iterate through number of place cells in place cells list
        for counter, cell in enumerate(numberCells):

            cellIndexes = np.where(spikeClusters == cell) # find the indices of the spikes of that cell
            cellTimestampsOE = spikeTimes[cellIndexes] # extract times at which that cell fired

            # need to convert times from Openephys times to seconds or milliseconds
            cellTimestamps = cellTimestampsOE / 30000 # divide by sampling rate to get seconds

            # Extract spikes from that session (spikeTimes and spikeClusters has all the data from that day's recording: VR1, sleep, VR2)
            cellTimesSession = cellTimestamps[(cellTimestamps >= startTime) & (cellTimestamps <= endTime)] # extract spikes from that session
            # then allocate 1s in the matrix in the timebins when it fired - use histogram function? 
            binnedSpikes, _= np.histogram(cellTimesSession, bins = timeBins)
            spikeMatrix[counter,:] = binnedSpikes # fill the matrix with the binned spikes
            zBinnedSpikes = stats.zscore(binnedSpikes) # z-score the binned spikes
            cellsTime[counter, :] = zBinnedSpikes # fill the matrix with the z-scored binned spikes

        # Remove rows with all NaNs
        cellsTimeNew = cellsTime
        numberCellsNew = numberCells
        row = 0
        for it in range(cellsTime.shape[0]):
            if np.isnan(cellsTimeNew[row,:]).all() == True:
                cellsTimeNew = np.delete(cellsTimeNew, row, 0)
                numberCellsNew = np.delete(numberCellsNew, row, 0)
                row = row
            else:
                row = row + 1
        
        # 5. ESTIMATE NUMBER OF COACTIVATION PATTERNS (= SIGNIFICANT EIGENVALUES) ------------------------------------------------------------------------------------------------------
        #  Create covariance (= correlation) matrix of the spike matrix.
        covMatrix = np.matmul(cellsTimeNew,np.transpose(cellsTimeNew)) / cellsTimeNew.shape[1]
        covMatrixNew = covMatrix
        row = 0
        for it in range(covMatrix.shape[0]):
            if np.isnan(covMatrix[row,:]).all() == True:   
                covMatrixNew = np.delete(covMatrixNew, row, 0)
                covMatrixNew = np.delete(covMatrixNew, row, 1)
                row = row
            else: 
                row = row + 1

        # Find number of total eigenvalues
        eigvals = np.linalg.eigvalsh(covMatrixNew)

        # Obtain number of significant eigenvalues based on Marcenko-Pastur distribution
        q = cellsTimeNew.shape[1] / cellsTimeNew.shape[0] # compute number of columns divided by number of rows
        s = 1 #cellsTimeNew.var() # get variance of matrix, should be 1 bc of normalization
        boundMax = s * ((1 + math.sqrt(1/q))**2) # find upper and lower bounds
        boundMin = s * ((1 - math.sqrt(1/q))**2)
        sigEigVals = np.where(eigvals > boundMax) # find eigenvalues above upper bound
        numberAssemblies = len(sigEigVals[0])
        print("number of estimated assemblies :" + str(len(sigEigVals[0])) + " out of " + str(len(eigvals)) + " eigenvalues")

        if numberAssemblies == 0: # if there are no coactivation patterns, move to next session
            continue
        
        # 6. EXTRACT ASSEMBLY PATTERNS: perform ICA -----------------------------------------------------------

        # then compute the independent components through the fastICA algorithm
        ICA = FastICA(n_components=numberAssemblies, max_iter=2000)
        weights = ICA.fit_transform(cellsTimeNew) 

        # this step is exclusive to Van de Ven 2016: they scale to unit length the weight vector and set the set sign of vector so that highest absolute weight is positive
        for pat in range(weights.shape[1]):
            weights[:,pat] = weights[:,pat] / np.linalg.norm(weights[:,pat])
            if np.max(np.absolute(weights[:,pat])) == np.max(weights[:,pat]):
                weights[:,pat] = weights[:,pat] # keep the sign of vector if highest absolute weight is positive
            else:
                weights[:,pat] = -(weights[:,pat]) # change the sign of vector if highest absolute weight is negative

        # 7. FIND PRINCIPAL CELLS OF EACH ASSEMBLY -------------------------------------------------------------------
        # iterate over weight vectors
        principalCells = [] # create empty list
        for pat in range(weights.shape[1]):
            # find the mean and std of those weights
            meanWeights = np.mean(weights[:,pat])
            stdWeights = np.std(weights[:,pat])
            # find the index of cells with weights 2 std above the mean = principal cells
            principalCells.append(np.where(weights[:,pat] > (meanWeights + 2*stdWeights)))

        # 8. FIND THE RATIO OF DISAMBIGUATING CELLS AMONG THE PRINCIPAL CELLS OF EACH ASSEMBLY (Do disambiguating cells tend to fire all together or are they firing with non-disambiguating ones?)----------------------
        # load disambiguating cells and place cells IDs for that session
        print('getting disambiguating cells ratio in cell assemblies')
        disCells = np.load(directory + filepath + "/Analysis/forPython/disambiguatingCells_session" + str(concatTable['sessions'][day]) + ".npy")
        placeCells = np.load(directory + filepath + "/Analysis/forPython/placeCells_session" + str(concatTable['sessions'][day]) + ".npy")

        # find the ratio of disambiguating cells among the principal cells of each assembly
        ratioDisCells = np.zeros(len(principalCells))
        for pat in range(len(principalCells)):
            # get the cell ID of the principal cells
            principalID = numberCellsNew[principalCells[pat]] 
            # find the number of disambiguating cells among the principal cells
            disCellsAmongPrincipal = np.where(principalID == disCells)
            if disCellsAmongPrincipal[0].size == 0:
                ratioDisCells[pat] = 0
            else:
                ratioDisCells[pat] = disCellsAmongPrincipal[0].shape[0] / principalID.shape[0]
        
        # Now do a control - Monte Carlo Simulation: ratio of randomly selected cells (Monte-Carlo) among the principal cells of each assembly
        shuffleNum = 1000
        randomPlaceCells = []
        ratioRandomPlaceCells = np.empty((len(principalCells), shuffleNum))
        ratioRandomPlaceCells[:] = np.nan
        for shuf in range(shuffleNum): # do a 1000 shuffles
            # do control for disambiguating cells, but use only place cells
            sample = random.sample(range(0, placeCells.shape[1]), disCells.shape[0])
            randomPlaceCells.append(placeCells[0,sample])
            for pat in range(len(principalCells)):
                principalID = numberCellsNew[principalCells[pat]]
                placeCellsAmongPrincipal = np.where(principalID == randomPlaceCells[shuf])
                if placeCellsAmongPrincipal[0].size == 0:
                    ratioRandomPlaceCells[pat,shuf] = 0
                else:
                    ratioRandomPlaceCells[pat,shuf] = placeCellsAmongPrincipal[0].shape[0] / principalID.shape[0]
    
                
        # 9. STORE WEIGHTS --------------------------------------------------------------------------------------------------------------------
        # store weights in a dataframe
        weightsDF = pd.DataFrame(weights)
        weightsDF.to_csv(directory + filepath + "/Analysis/forPython/weights" + str(concatTable['sessions'][day]) + "v2.csv", index = False) 
        np.save(directory + filepath + "/Analysis/forPython/principalCells" + str(concatTable['sessions'][day]) + "v2.npy", principalCells)
        np.save(directory + filepath + "/Analysis/forPython/ratioDisCells" + str(concatTable['sessions'][day]) + "v2.npy", ratioDisCells)  
        np.save(directory + filepath + "/Analysis/forPython/ratioRandomPlaceCells" + str(concatTable['sessions'][day]) + "v2.npy", ratioRandomPlaceCells)
