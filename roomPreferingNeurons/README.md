## Inputs:

The scripts in this folder require certain variables to be computed beforehand: 
- ratemapPerTrial.npy: 2-D numpy array, binned firing rate (spikes/second) of each cell, number of rows = number of trials and number of columns = number of spatial bins. 
- speedPerTrial.npy: 2-D numpy array, binned speed,  number of rows = number of trials, number of columns = number of spatial bins. 
- posPF.npy: 1-D numpy array, binary vector of length = number of spatial bins. 1 = location of place field. 0 = outside of place field. 

## Outputs: 

The scripts in this folder output the following file: 
- significantRatemaps.npy: 2-D numpy array, binned firing rate (spikes/second) of all room-prefering cells, 
    number of rows = number of room-prefering cells and number of columns = number of spatial bins. 
- significantCells.npy: 1-D numpy array, vector with cell ID of room-prefering cells. Length = number of room-prefering cells. 
