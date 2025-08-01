# Script to generate the rewards and losses
import matplotlib.pyplot as plt
import numpy as np


def rewards_hidden_states(whatRewards, baseNo):
    """
    Calculate rewards based on the specified reward type and a base value.

    Parameters:
    whatRewards (str): The type of rewards to calculate. 
                       Options include:
                       - "loss_incorrect"
                       - "loss_correct"
                       - "gain_incorrect"
                       - "gain_correct"
    baseNo (float): The base value used to calculate the rewards.

    Returns:
    tuple: A tuple containing:
        - allRewards (numpy.ndarray): An array of calculated rewards based on the specified type.
        - ratioIdx (list): A list of indices used to organize the x-axis of figures based on the ratio.
    """
    if whatRewards=="loss_incorrect":

        granRewards = 0.25 #...granuality of ratios
        minRatio = 0.25
        maxRatio = 2.25
        ratioRange = np.arange(minRatio,maxRatio,granRewards)
        ratioRange = np.array([0.125,0.25,0.5,1,2,4,8])
        ratioRange = np.transpose([ratioRange])
        noRewards = ratioRange.shape[0]

        baseRewards = np.transpose(np.array([baseNo,-baseNo,-baseNo,baseNo]))
        allRewards = np.tile(baseRewards, (noRewards,1))#make multiples of these
        allRewards = allRewards.astype('float64')
        allRewards[:,[2]] *=ratioRange #...transform the loss values based on the ratio array
        
        ratioIdx = [2,1]#...this will be used to organize x-axis of figures so they show as the ratio increases,
        #..so this gives the numerator and operator
        
    if whatRewards=="loss_correct":

        granRewards = 0.25 #...granuality of ratios
        minRatio = 0.25
        maxRatio = 2.25
        ratioRange = np.arange(minRatio,maxRatio,granRewards)
        ratioRange = np.array([0.125,0.25,0.5,1,2,4,8])
        ratioRange = np.transpose([ratioRange])
        noRewards = ratioRange.shape[0]

        baseRewards = np.transpose(np.array([baseNo,-baseNo,-baseNo,baseNo]))
        allRewards = np.tile(baseRewards, (noRewards,1))#make multiples of these
        allRewards = allRewards.astype('float64')
        allRewards[:,[1]] *=ratioRange #...transform the loss values based on the ratio array
        
        ratioIdx = [1,2]#...this will be used to organize x-axis of figures so they show as the ratio increases,
        #..so this gives the numerator and operator
        
    if whatRewards=="gain_incorrect":

        granRewards = 0.25 #...granuality of ratios
        minRatio = 0.25
        maxRatio = 2.25
        ratioRange = np.arange(minRatio,maxRatio,granRewards)
        ratioRange = np.array([0.125,0.25,0.5,1,2,4,8])
        ratioRange = np.transpose([ratioRange])
        noRewards = ratioRange.shape[0]

        baseRewards = np.transpose(np.array([baseNo,-baseNo,-baseNo,baseNo]))
        allRewards = np.tile(baseRewards, (noRewards,1))#make multiples of these
        allRewards = allRewards.astype('float64')
        allRewards[:,[0]] *=ratioRange #...transform the loss values based on the ratio array
        
        ratioIdx = [0,3]#...this will be used to organize x-axis of figures so they show as the ratio increases,
        #..so this gives the numerator and operator
        
    if whatRewards=="gain_correct":

        granRewards = 0.25 #...granuality of ratios
        minRatio = 0.25
        maxRatio = 2.25
        ratioRange = np.arange(minRatio,maxRatio,granRewards)
        ratioRange = np.array([0.125,0.25,0.5,1,2,4,8])
        ratioRange = np.transpose([ratioRange])
        noRewards = ratioRange.shape[0]

        baseRewards = np.transpose(np.array([baseNo,-baseNo,-baseNo,baseNo]))
        allRewards = np.tile(baseRewards, (noRewards,1))#make multiples of these
        allRewards = allRewards.astype('float64')
        allRewards[:,[3]] *=ratioRange #...transform the loss values based on the ratio array
        
        ratioIdx = [3,0]#...this will be used to organize x-axis of figures so they show as the ratio increases,
        #..so this gives the numerator and operator
        
        
    return allRewards,ratioIdx

