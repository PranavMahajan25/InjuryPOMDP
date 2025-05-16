import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import stats
import copy
import importlib

from convenience_functions_rumPOMDP import *

plt.rcParams.update({'font.size': 22})

def qvalues_figure(self,rewardID,figSave):

    beliefRange = self.beliefRange

    fig,axs = plt.subplots(1,len(rewardID))
    fig.set_figheight(8)
    fig.set_figwidth(8*len(rewardID))
    lines = [":","dashed","-"]
    lineColors = ["#FEC400", "#40E0D0"]

    # plt.rcParams.update({'font.size': 18})
    
    yLims = [-20,100]

    for r,reward in enumerate(rewardID):            
        qVals = self.actionValues[reward,:,:]
    
        if len(rewardID)==1:
            axsr = axs
        else:
            axsr = axs[r]

        if self.decisionThreshold[r,0] < self.decisionThreshold[r,2]:
            thresh1 = self.decisionThreshold[r,0]
            axsr.plot([self.decisionThreshold[r,0],self.decisionThreshold[r,0]],yLims,'k', linestyle="dashed",linewidth=3)
        else:
            thresh1 = self.decisionThreshold[r,2]
            axsr.plot([self.decisionThreshold[r,2],self.decisionThreshold[r,2]],yLims,'k', linestyle="dashed",linewidth=3)   
        if self.decisionThreshold[r,1] > self.decisionThreshold[r,3]:
            thresh2 = self.decisionThreshold[r,1]
            axsr.plot([self.decisionThreshold[r,1],self.decisionThreshold[r,1]],yLims,'k', linestyle="dashed",linewidth=3)
        else:
            thresh2 = self.decisionThreshold[r,3]
            axsr.plot([self.decisionThreshold[r,3],self.decisionThreshold[r,3]],yLims,'k', linestyle="dashed",linewidth=3)    
        # axsr.plot([self.decisionThreshold[r,4],self.decisionThreshold[r,4]],yLims,'k',linewidth=1)

        actionValues = self.actionValues[reward,:,:]
        # axs[0].set_title(self.name)
        axsr.plot([0,1],[0,0],color = 'k', linestyle = '-')
        axsr.plot(beliefRange,actionValues[:,0],'#5E5EDA',linewidth=5, label='$a_{act}$')
        axsr.plot(beliefRange,actionValues[:,1],'#DC267F',linewidth=5, label='$a_{r&r}$')

        A3sampleValues = self.actionValues[reward,:,2]
        A4sampleValues = self.actionValues[reward,:,3]
        axsr.plot(beliefRange,A3sampleValues,lineColors[0],linewidth=5, label="$a_{que}$") #WILL ONLY PLOT ONE IN FIRST 
        axsr.plot(beliefRange,A4sampleValues,lineColors[1],linewidth=5, label="$a_{nul}$") 


        # axsr.set(xlim = [0,1],xticks = [0,0.5,1],xlabel ='Belief State $B$')
        axsr.set(xlim = [0,1],xticks = [0,0.5,1])
        axsr.set(ylim = yLims)

        axsr.spines['left'].set_linewidth(3)  
        axsr.spines['bottom'].set_linewidth(3)
        axsr.spines['right'].set_linewidth(3)  
        axsr.spines['top'].set_linewidth(3)  

        # if r == 0:
        #     axsr.set(ylabel='Value of Action')
        
        axsr.legend()
              

    if (figSave != ""): #...print individual to file

        # create the full path for the subfolder
        subPath = os.path.join(figSave, "qvalues/")

        # check if the subfolder already exists
        if not os.path.exists(subPath):
        # create the subfolder if it does not exist
            os.makedirs(subPath)

        # plt.savefig(subPath+'qvalues.pdf')
        plt.savefig(subPath+'qvalues.png', dpi=1500)
        # plt.savefig(subPath+'qvalues'+str(self.name)+'.svg')
        plt.close(fig)

    return thresh1, thresh2

