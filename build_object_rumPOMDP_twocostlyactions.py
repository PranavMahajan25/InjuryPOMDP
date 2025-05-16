#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### HOUSEKEEPING
#import what is necessary and some things that may be unnecessary, and convenience functions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import stats
import copy
import importlib

from modelFunctions_rumPOMDP import *
from convenience_functions_rumPOMDP import *

class POMDP:
    
    def __init__(self,xObvs,obvStep):
        #...set the parameters for the observation distributions
        A3xObvs = xObvs[0]
        A4xObvs = xObvs[1]

        ### A3 margEvid and obvPrior
        self.A3name = A3xObvs[0]
        self.A3xMeans = np.array(A3xObvs[1:3])
        self.A3xStd = np.array(A3xObvs[3:5])
        self.A3obvs = np.arange(obvStep,100,obvStep) 
        
        A3stateNo = self.A3xMeans.shape[0]
        A3obvPrior = np.zeros([A3stateNo,len(self.A3obvs)]) #...priors for all observations (typically set at 0)
        
        beliefD = 0.01 #...discretize the belief space
        beliefRange = np.arange(0,1+beliefD,beliefD) #...generate all possible values of b, given the discretization
        
        for x in range(A3stateNo): 

            A3obvMean = self.A3xMeans[x]
            A3obvStd = self.A3xStd[x]

            A3obvPrior[x,:] = stats.norm.pdf(self.A3obvs,A3obvMean,A3obvStd)*obvStep
           
        self.A3margEvid,self.A3posterior = bayes_theorem(beliefRange,A3obvPrior,'-')

        A3obvPrior[A3obvPrior<0.001] = 0.001#...do not allow non-zero observation priors
        
        #...normalize each row so the sum of belief states is 1
        row_sums = A3obvPrior.sum(axis=1)
        A3obvPrior = A3obvPrior / row_sums[:, np.newaxis]
        
        ### A4 margEvid and obvPrior
        self.A4name = A4xObvs[0]
        self.A4xMeans = np.array(A4xObvs[1:3])
        self.A4xStd = np.array(A4xObvs[3:5])
        self.A4obvs = np.arange(obvStep,100,obvStep) 
        
        A4stateNo = self.A4xMeans.shape[0]
        A4obvPrior = np.zeros([A4stateNo,len(self.A4obvs)]) #...priors for all observations (typically set at 0)
        
        beliefD = 0.01 #...discretize the belief space
        beliefRange = np.arange(0,1+beliefD,beliefD) #...generate all possible values of b, given the discretization
        
        for x in range(A4stateNo): 

            A4obvMean = self.A4xMeans[x]
            A4obvStd = self.A4xStd[x]

            A4obvPrior[x,:] = stats.norm.pdf(self.A4obvs,A4obvMean,A4obvStd)*obvStep
           
        self.A4margEvid,self.A4posterior = bayes_theorem(beliefRange,A4obvPrior,'-')

        A4obvPrior[A4obvPrior<0.001] = 0.001#...do not allow non-zero observation priors
        
        #...normalize each row so the sum of belief states is 1
        row_sums = A4obvPrior.sum(axis=1)
        A4obvPrior = A4obvPrior / row_sums[:, np.newaxis]

        self.beliefRange = beliefRange
        self.obvStep = obvStep

        # return
        self.A3obvPrior = A3obvPrior
        self.A4obvPrior = A4obvPrior

        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### calculate the belief transition matrix
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 
    def belief_transition_matrix(self):
        self.A3beliefTrans = np.zeros([len(self.beliefRange),len(self.beliefRange)])
        self.A4beliefTrans = np.zeros([len(self.beliefRange),len(self.beliefRange)])
        
        ### A3 belief transition matrix
        for b,belief in enumerate(self.beliefRange):
            
            A3beliefTrans = self.A3beliefTrans[b,:] #...transition vector for P(B'|B_i)

            _,bPrimeRange = bayes_theorem(belief,self.A3obvPrior,'-') #...for each belief and each possible cue, find all probabilities
            bPrimeRange = bPrimeRange[0]
            
            for bp,bPrime in enumerate(bPrimeRange): #...tmp is a vector of all possible B' for B_i              
                bFind = index_finder(self.beliefRange,bPrime) #...returns the closest B to the B' being iterated through
                
                A3beliefTrans[bFind] = A3beliefTrans[bFind]+self.A3margEvid[b,bp] 
               
            self.A3beliefTrans[b,:] = A3beliefTrans
        
        ### A4 belief transition matrix
        for b,belief in enumerate(self.beliefRange):
            
            A4beliefTrans = self.A4beliefTrans[b,:] #...transition vector for P(B'|B_i)

            _,bPrimeRange = bayes_theorem(belief,self.A4obvPrior,'-') #...for each belief and each possible cue, find all probabilities
            bPrimeRange = bPrimeRange[0]
            
            for bp,bPrime in enumerate(bPrimeRange): #...tmp is a vector of all possible B' for B_i              
                bFind = index_finder(self.beliefRange,bPrime) #...returns the closest B to the B' being iterated through
                
                A4beliefTrans[bFind] = A4beliefTrans[bFind]+self.A4margEvid[b,bp] 
               
            self.A4beliefTrans[b,:] = A4beliefTrans
              
    def belief_transition_matrix_plot(self,figSave):
            ### A3 plot
            fig, ax = plt.subplots(1,2)
            fig.set_figheight(5)
            fig.set_figwidth(5*3)

            ax[0].plot(self.A3obvs,np.transpose(self.A3obvPrior),linewidth=8)
            ax[0].set(ylim=[0,0.1],xlim=[0,100],title=self.A3name,xlabel='Observation estimate', ylabel='Probability')
            ax[0].set_aspect('equal', adjustable='box')
            ax[0].set_aspect(1.0/ax[0].get_data_ratio(), adjustable='box')

            im = ax[1].imshow(self.A3beliefTrans, cmap='copper', aspect = 'auto',vmin = 0,vmax = 0.04)
            ax[1].set(xlabel='B\'',ylabel='B')
            fig.colorbar(im, ax=ax[1], label='P(B\'|B)')
            ax[1].set_aspect(1.0/ax[1].get_data_ratio(), adjustable='box')
            plt.tight_layout()

            if figSave != "": #...print individual to file

                # create the full path for the subfolder
                subPath = os.path.join(figSave, "observation_distributions/")

                # create the subfolder if it does not exist
                if not os.path.exists(subPath):
                    os.makedirs(subPath)

                # plt.savefig(subPath+'belief_transition'+str(self.A3name)+'.pdf')
                plt.savefig(subPath+'belief_transition'+str(self.A3name)+'.png', dpi=1000)    
          #      plt.savefig(subPath+'belief_transition'+str(self.name)+'.svg')
                plt.close(fig)

            ### A4 plot
            fig, ax = plt.subplots(1,2)
            fig.set_figheight(5)
            fig.set_figwidth(5*3)

            ax[0].plot(self.A4obvs,np.transpose(self.A4obvPrior),linewidth=8)
            ax[0].set(ylim=[0,0.1],xlim=[0,100],title=self.A4name,xlabel='Observation estimate', ylabel='Probability')
            ax[0].set_aspect('equal', adjustable='box')
            ax[0].set_aspect(1.0/ax[0].get_data_ratio(), adjustable='box')

            im = ax[1].imshow(self.A4beliefTrans, cmap='copper', aspect = 'auto',vmin = 0,vmax = 0.04)
            ax[1].set(xlabel='B\'',ylabel='B')
            fig.colorbar(im, ax=ax[1], label='P(B\'|B)')
            ax[1].set_aspect(1.0/ax[1].get_data_ratio(), adjustable='box')
            plt.tight_layout()

            if figSave != "": #...print individual to file

                # create the full path for the subfolder
                subPath = os.path.join(figSave, "observation_distributions/")

                # create the subfolder if it does not exist
                if not os.path.exists(subPath):
                    os.makedirs(subPath)

                # plt.savefig(subPath+'belief_transition'+str(self.A4name)+'.pdf')
                plt.savefig(subPath+'belief_transition'+str(self.A4name)+'.png', dpi=1500)
                plt.close(fig)
                
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## calculate the value of each state using dynamic programming
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def value_iteration(self,allRewards, A3Cost, A4Cost,figPrint,figSave):
        
        beliefRange = self.beliefRange
        A3beliefTrans = self.A3beliefTrans
        A4beliefTrans = self.A4beliefTrans
        
        self.beliefValues = np.full([allRewards.shape[0],len(beliefRange)],np.nan)
        self.actionValues = np.full([allRewards.shape[0],len(beliefRange),4],np.nan)
        self.decisionThreshold = np.full([allRewards.shape[0],5],np.nan)
        
        self.allRewards = allRewards
        
        if figPrint :
            fig, ax = plt.subplots(2,allRewards.shape[0])
            fig.set_figheight(5*2)
            fig.set_figwidth((5*allRewards.shape[0]))

        for r,rewards in enumerate(allRewards):
            beliefValues = np.zeros([len(beliefRange)])
            actionValues = np.zeros([len(beliefRange),4])
            
            vDelta = 1 #...threshold for adjusting the value in an iteration
            qDelta = 1 #...RB
            
            updateTrue = True
            
            plotValue = []
            plotAction = []
            k = 0 #...this will iterate as the values converge to within the vDelta amount
            
            while updateTrue:    
                
                v = copy.copy(beliefValues) #...vector of all the belief values which is iterated on each k


                for b,belief in enumerate(beliefRange): #...for each belief

                    A3_bPrime = A3beliefTrans[b,:] #...vector of state transition probabilities for some belief state
                    A4_bPrime = A4beliefTrans[b,:] #...vector of state transition probabilities for some belief state
                    
                    #...functions for calculating all q values
                    # qA3Sample = qVal_cost(A3Cost,A3_bPrime,v,self.obvStep,'-') 
                    # qA4Sample = qVal_cost(A4Cost,A4_bPrime,v,self.obvStep,'-') 
                    
                    # new edits after inputs from Peter
                    qA3Sample = qVal_cost_belief_dependent_a_que(A3Cost,belief,A3_bPrime,v,self.obvStep,'-') 
                    qA4Sample = qVal_cost(A4Cost,A4_bPrime,v,self.obvStep,'-') 

                    qL,qR,_ = qVal_lr(rewards,belief,'-') 
                    
                    # print(np.array([qL,qR,qA3Sample[0],qA4Sample[0]]))
                    # actionValues[b,:] = [qL,qR,qA3Sample,qA4Sample]
                    actionValues[b,:] = np.array([qL,qR,qA3Sample[0],qA4Sample[0]])
                    
                    beliefValues[b] = belief_values(actionValues[b,:])

                #...when the maximum difference between any of the belief values on each iterate is more than vDelta,
                #then continue to iterate until this is not true.
                updateTrue = np.max(abs(v-beliefValues))>vDelta 
                
                plotValue = np.append(plotValue,beliefValues)
                plotAction = np.append(plotAction,actionValues[:,2])
                
                k = k+1 #...keep counting the iterations!
                
                #...save the final belief values and action values
                self.beliefValues[r,:] = beliefValues
                self.actionValues[r,:,:] = actionValues
                
                #...save the decision thresholds for each reward function and cost
                self.decisionThreshold[r,0] = beliefRange[index_finder(actionValues[:,2],actionValues[:,0])]
                self.decisionThreshold[r,1] = beliefRange[index_finder(actionValues[:,2],actionValues[:,1])] 
                self.decisionThreshold[r,2] = beliefRange[index_finder(actionValues[:,3],actionValues[:,0])]
                self.decisionThreshold[r,3] = beliefRange[index_finder(actionValues[:,3],actionValues[:,1])]  
                self.decisionThreshold[r,4] = beliefRange[index_finder(actionValues[:,2],actionValues[:,3])] 
                
                if figPrint:

                    plotValue = np.resize(np.transpose(plotValue),(k,len(beliefRange)))
                    plotAction = np.resize(np.transpose(plotAction),(k,len(beliefRange)))
                    
                    # print(plotValue.shape, plotAction.shape)
                    if allRewards.shape[0] == 1:
                        im = ax[0].imshow(plotValue, cmap='gray', aspect = 'auto')
                        im = ax[1].imshow(plotAction, cmap='gray', aspect = 'auto')
                        # ax[0].set(xlim = [0,100],xticks = [0,0.5,1],xlabel ='Belief State $B$',ylabel='k for V(B)')
                        # ax[1].set(xlim = [0,100],xticks = [0,0.5,1],xlabel ='Belief State $B$',ylabel='k for Q(Sample)')
                        ax[0].set(xlim = [0,100], xticks = [0,49,99], xticklabels = [0, 0.5, 1])
                        ax[1].set(xlim = [0,100], xticks = [0,49,99], xticklabels = [0, 0.5, 1])
                    else:
                        im = ax[0,r].imshow(plotValue, cmap='gray', aspect = 'auto')
                        im = ax[1,r].imshow(plotAction, cmap='gray', aspect = 'auto')
                        # ax[0,r].set(xlim = [0,100],xticks = [0,0.5,1],xlabel ='Belief State $B$',ylabel='k for V(B)')
                        # ax[1,r].set(xlim = [0,100],xticks = [0,0.5,1],xlabel ='Belief State $B$',ylabel='k for Q(Sample)')
                        ax[0,r].set(xlim = [0,100], xticks = [0,49,99], xticklabels = [0, 0.5, 1])
                        ax[1,r].set(xlim = [0,100], xticks = [0,49,99], xticklabels = [0, 0.5, 1])
                    # if r == 0:
                        # im = ax[0,r].set_title(self.name)

            if (figSave != ""): #...print individual to file

                # create the full path for the subfolder
                subPath = os.path.join(figSave, "value_iteration/")

                # check if the subfolder already exists
                if not os.path.exists(subPath):
                # create the subfolder if it does not exist
                    os.makedirs(subPath)

                # plt.savefig(subPath+'value_iteration.pdf')
                plt.savefig(subPath+'value_iteration.png', dpi=1500)
            #       plt.savefig(subPath+'value_iteration'+str(self.name)+'.svg')
                plt.close(fig)


    
