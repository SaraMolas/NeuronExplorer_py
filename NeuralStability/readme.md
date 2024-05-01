# Inputs:

The scripts in this folder require certain variables to be computed beforehand: 
- ratemapsTrials.npy: 3D-numpy array where axis 0 corresponds to cell number, axis 1 corresponds to number of spatial bins, and axis 2 corresponds to number of trials.
- cellLabels.npy: 2D-numpy array where row corresponds to cell number, column 1 is the cell ID number and column 2 is a binary vector indicating if a neuron has been classified as a place cell (True) or not (False). 
- numberLicks.npy: vector with the total number of licks recorded in each session. vector length = number of sessions.
- runningSpeed.npy: vector with the average running speed (in cm/s) recorded in each session. vector length = number of sessions.
- numberTrials.npy: vector with the total number of trials recorded in each session. vector length = number of sessions.
- pausingTime.npy: vector with the total pausing time (in seconds) recorded in each session. vector length = number of sessions.
- preRewLicks.npy: vector with the total number of licks recorded in the pre-reward area in each session. vector length = number of sessions.
- preRewSlope.npy: vector with the deceleration slope in the pre-reward area in each session. vector length = number of sessions.
- roomCLicks.npy: vector with the total number of licks recorded in room C in each session. vector length = number of sessions.
- roomCspeed.npy: vector with the average running speed (in cm/s) recorded in room C in each session. vector length = number of sessions.
- PCstability.npy: vector with the average stability of all recorded place cells in each session. vector length = number of sessions.

# Outputs:

The scripts in this folder output the following file: 
- stabilityScores.npy: vector containing the stability scores for all recorded neurons in each session. If a neuron was not classified as a place cell, then the stability score will be NaN. 
