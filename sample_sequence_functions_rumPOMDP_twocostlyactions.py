import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import stats

import importlib

from modelFunctions_rumPOMDP import *
from convenience_functions_rumPOMDP import *

def sample_sequence_observation_accumulated_cost(self, rewardID, beliefStart, maxSample, noTrials, obv_mismatch, obv_truestate=1, epsilon=0, biased_action=2):
    """
    Samples a sequence of observations and accumulates costs based on the chosen actions and beliefs.

    Parameters:
    - rewardID (list): A list of reward identifiers used to index into belief and action values.
    - beliefStart (float): The initial belief state from which to start sampling.
    - maxSample (int): The maximum number of samples to draw in each trial.
    - noTrials (int): The number of trials to run.
    - obv_mismatch (str): Indicates whether to use mismatched observations ('mismatch' or other).
    - obv_truestate (int, optional): The true state of the observation. Default is 1.
    - epsilon (float, optional): The probability of choosing a biased action. Default is 0.
    - biased_action (int, optional): The action to take when biased. Default is 2.

    Returns:
    - None: The function modifies the instance's attributes to store results of the sampling process.
    
    Attributes modified:
    - self.noSamples_t: Array storing the number of samples taken before termination for each trial.
    - self.termChoice_t: Array storing the choice made at the end of each trial.
    - self.posterior_beliefs: Array storing the posterior beliefs at each time step.
    - self.posterior_beliefs_ending_in_A1: List of posterior beliefs ending in action A1.
    - self.posterior_beliefs_ending_in_A2: List of posterior beliefs ending in action A2.
    - self.noA3: Array counting the number of times action A3 was chosen.
    - self.noA4: Array counting the number of times action A4 was chosen.
    """
    # Initialize arrays to store results
    # avTermChoice = np.full([len(rewardID)],np.nan)
    # avTermCost = np.full([len(rewardID)],np.nan)
    self.noSamples_t = np.full([len(rewardID),noTrials],np.nan)
    self.termChoice_t = np.full([len(rewardID),noTrials],np.nan)
    self.noSamples_av = np.full([len(rewardID),3],np.nan)
    self.noA3 = np.full([noTrials],np.nan)
    self.noA4 = np.full([noTrials],np.nan)
    self.posterior_beliefs = np.zeros([noTrials, maxSample]) 
    self.posterior_beliefs_ending_in_A1 = []
    self.posterior_beliefs_ending_in_A2 = []

    for r,rewards in enumerate(rewardID):                
        beliefValues = self.beliefValues[rewards,:]
        actionValues = self.actionValues[rewards,:,:]
        beliefRange = self.beliefRange
        
        termChoice = np.full([noTrials],np.nan)
        termCost = np.full([noTrials],np.nan)
        termQ = np.full([noTrials],np.nan)

        for t in range(noTrials):
            
            if obv_mismatch=="mismatch":
            #    print('mismatching std or means')
                A3obvSequence = np.random.normal(loc=self.A3xMeans_mismatch[obv_truestate],scale=self.A3xStd_mismatch[obv_truestate],size=maxSample)
                A4obvSequence = np.random.normal(loc=self.A4xMeans_mismatch[obv_truestate],scale=self.A4xStd_mismatch[obv_truestate],size=maxSample)
                #  print(self.xStd_mismatch[1])

            else:
                #   print('std and means as standard')
                A3obvSequence = np.random.normal(loc=self.A3xMeans[obv_truestate], scale=self.A3xStd[obv_truestate], size=maxSample)
                A4obvSequence = np.random.normal(loc=self.A4xMeans[obv_truestate], scale=self.A4xStd[obv_truestate], size=maxSample)
            
            # print(self.A3xMeans, self.A3xStd, A3obvSequence)
            # print(self.A4xMeans, self.A4xStd, A4obvSequence)

            qVals_samp = np.zeros([maxSample,4])

            prior_samp = np.zeros([maxSample]) 
            posterior_samp = np.zeros([maxSample])

            margEvid_samp = np.zeros([maxSample])

            choiceProbs_samp = np.zeros([maxSample,4])
            choiceMade_samp = np.zeros([maxSample])

            idealChoice_samp = np.zeros([maxSample])

            samp = 0
            prior_samp[samp] = beliefStart
            posterior_samp[samp] = beliefStart 
            
            sampling = True
            maxSample_below = True
            
            while sampling and maxSample_below:
            
                #get the qvalues
                ibelief = index_finder(beliefRange,prior_samp[samp])
                qVals_samp[samp,:] = actionValues[ibelief,:] 
                
                #get the best choice
                getWhere = np.where(qVals_samp[samp,:]==np.max(qVals_samp[samp,:]))
                idealChoice_samp[samp] = getWhere[0][0]

                # add any biases here
                if np.random.uniform() < epsilon:
                    choiceMade_samp[samp] = biased_action
                else:
                    choiceMade_samp[samp] = idealChoice_samp[samp]

                #end if the choice is to stop sampling
                sampling= (choiceMade_samp[samp]==2) or (choiceMade_samp[samp]==3) #keep running if it is 2 or 3 (sample action)
                
                maxSample_below = samp<maxSample #keep running if t is less than maxSample

                if not sampling: #...end if they chose a terminating action
                    break

                #update the belief for the loop
                if choiceMade_samp[samp] == 2: #A3
                    iobv = index_finder(self.A3obvs,A3obvSequence[samp])
                
                    priorsX = self.A3obvPrior[:,iobv]
                    priorsX = np.reshape(priorsX,(2,1))
                    _,posterior_samp[samp]  = bayes_theorem(prior_samp[samp],priorsX,'-')

                elif choiceMade_samp[samp] == 3: #A4
                    iobv = index_finder(self.A4obvs,A4obvSequence[samp])
                
                    priorsX = self.A4obvPrior[:,iobv]
                    priorsX = np.reshape(priorsX,(2,1))
                    _,posterior_samp[samp]  = bayes_theorem(prior_samp[samp],priorsX,'-')
                else:
                    print("Error in choice made")
                
                maxSample_below = samp<(maxSample-1) #...if there 
                
                if not maxSample_below:
                    break
                    
                prior_samp[samp+1] = posterior_samp[samp]
                # print(posterior_samp[samp])
                samp = samp +1

            self.noSamples_t[r,t] = samp #...for every trial, the number of samples before termination
            self.termChoice_t[r,t] = choiceMade_samp[samp]

            # ##### new addition
            self.posterior_beliefs[t, :] = prior_samp
            self.posterior_beliefs[t, samp:maxSample] = prior_samp[samp] 

            self.noA3[t] = np.sum(choiceMade_samp==2)
            self.noA4[t] = np.sum(choiceMade_samp==3)

            if choiceMade_samp[samp] == 0:
                self.posterior_beliefs_ending_in_A1.append(self.posterior_beliefs[t, :])
            elif choiceMade_samp[samp] == 1:
                self.posterior_beliefs_ending_in_A2.append(self.posterior_beliefs[t, :]) 
        
        #end of trials
       

        self.posterior_beliefs_ending_in_A1 = np.array(self.posterior_beliefs_ending_in_A1)
        self.posterior_beliefs_ending_in_A2 = np.array(self.posterior_beliefs_ending_in_A2)

        # print(self.noA3, self.noA4)
        # print(self.termChoice_t)
        # print(self.posterior_beliefs)


        # self.noSamples_av[r,0] = np.nanmean(self.noSamples_t[r,self.termChoice_t[r,:]==0]) 
        # self.noSamples_av[r,1] = np.nanmean(self.noSamples_t[r,self.termChoice_t[r,:]==1]) 
        
        # self.noSamples_av[r,2] = np.nanmean(self.noSamples_t[r,:]) #...average number of samples across kRuns


        
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ### Plotting code - Plot 2 shows a heatmap of sampling
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# def sample_sequence_figure(self,allRewards,allCosts,rewardID,costID,ratioidx,figSave):
                                                        
#     plotRewards = self.allRewards
#     rewardRatio = plotRewards[:,ratioidx[0]]/plotRewards[:,ratioidx[1]]
    
#     fig2,axs = plt.subplots(1,2, gridspec_kw={'width_ratios': [1,1]}, sharex=False, sharey=True)            
#     fig2.set_figheight(5)
#     fig2.set_figwidth(7)
    
#     plt.rcParams.update({'font.size': 18})

#     lines = [":","dashed","-"]

#     lineColors = ["#FE6100","#FEC400","#FEEB00",'black','black']
            
#     for action in range(2): 
                
#         xData = np.arange(0,len(rewardRatio),1)#rewardRatio
            
#         for c in range(len(costID)):
            
#             yData_1 = self.decisionThreshold[:,c,action]
            
#             axs[action].plot(xData,yData_1,color='black',linewidth = 1,linestyle=lines[c])
            
#         axs[action].set_title(self.name, fontsize=10)
               
#         axs[action].set_ylim([0,1])
#         axs[action].set_yticks([0,0.5,1])
        
#         axs[action].set_xlim([-1,len(xData)-0.5])            
#         axs[action].set_xticks(xData)                 
#         axs[action].set_xticklabels(rewardRatio)                
#         axs[action].tick_params(axis='x', rotation=45, width=4)
        
#         axs2 = axs[action].twinx()    
        
#         #run for the sampling
#         for c in range(len(costID)):     
            
#             yData_2 = self.noSamples_av[:,c,action]
#             axs2.plot(xData,yData_2,color=lineColors[action],linewidth = 3,linestyle=lines[c])

#         axs2.set_ylim([0,50])        

#         axs[action].spines['left'].set_linewidth(3)  
#         axs[action].spines['bottom'].set_linewidth(3)
#         axs[action].spines['right'].set_linewidth(3)  
#         axs[action].spines['top'].set_linewidth(3)  
        
#     plt.setp(axs[0], ylabel='Belief State')
#     plt.setp(axs[1], xlabel='Loss:Reward Ratio (logarithmic)')
#     plt.setp(axs2, ylabel='mean no. samples')
#     fig2.tight_layout()
    
#     if (figSave != ""): #...print individual to file

#         # create the full path for the subfolder
#         subPath = os.path.join(figSave, "sample_sequence/")

#         # check if the subfolder already exists
#         if not os.path.exists(subPath):
#         # create the subfolder if it does not exist
#             os.makedirs(subPath)

#         plt.savefig(subPath+'sample_seq_'+str(self.name)+'.pdf')
#         plt.savefig(subPath+'sample_seq'+str(self.name)+'.svg')


    