import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import random
from scipy import stats
import copy
import importlib

from convenience_functions_rumPOMDP import *


plt.rcParams.update({'font.size': 30})

def qvalues_figure(self,rewardID,costID,figSave):

    beliefRange = self.beliefRange

    fig,axs = plt.subplots(1,len(rewardID))
    fig.set_figheight(8)
    fig.set_figwidth(8*len(rewardID))
    lines = ["-"]
    # lines = [":","dashed","-"]
    lineColors = ["#FEC400","#FEC400","#FEC400"]
    # plt.rcParams.update({'font.size': 18})
    
    yLims = [-75,100]
    # yLims = [-20,100]

    for r,reward in enumerate(rewardID):
        
        for c,cost in enumerate(costID): #print the threshold lines
            
            qVals = self.actionValues[reward,cost,:,:]
            
            if len(rewardID)==1:
                axsr = axs
            else:
                axsr = axs[r]

            axsr.plot([self.decisionThreshold[r,c,0],self.decisionThreshold[r,c,0]],yLims,'k',linestyle="--",linewidth=3)
            axsr.plot([self.decisionThreshold[r,c,1],self.decisionThreshold[r,c,1]],yLims,'k',linestyle="--",linewidth=3)      
        
        actionValues = self.actionValues[reward,0,:,:]
        # axsr.set_title(self.name)
        axsr.plot([0,1],[0,0],color = 'k', linestyle = '-')
        axsr.plot(beliefRange,actionValues[:,0],'#5E5EDA',linewidth=5, label='$a_{act}$')
        axsr.plot(beliefRange,actionValues[:,1],'#DC267F',linewidth=5, label='$a_{r&r}$')

        

        for c,cost in enumerate(costID):  #then print the three costs

            sampleValues = self.actionValues[reward,cost,:,2]
            axsr.plot(beliefRange,sampleValues,lineColors[c],linestyle=lines[c],linewidth=5, label='$a_{que}$') #WILL ONLY PLOT ONE IN FIRST 

        # axsr.set(xlim = [0,1],xticks = [0,0.5,1],xlabel ='Belief State $B$')
        axsr.set(xlim = [0,1],xticks = [0,0.5,1])
        axsr.set(ylim = yLims)

        axsr.spines['left'].set_linewidth(3)  
        axsr.spines['bottom'].set_linewidth(3)
        axsr.spines['right'].set_linewidth(3)  
        axsr.spines['top'].set_linewidth(3)  

        axsr.legend()

        # if r == 0:
            # axsr.set(ylabel='Value of Action')
              

    if (figSave != ""): #...print individual to file

        # create the full path for the subfolder
        subPath = os.path.join(figSave, "qvalues/")

        # check if the subfolder already exists
        if not os.path.exists(subPath):
        # create the subfolder if it does not exist
            os.makedirs(subPath)

        # plt.savefig(subPath+'qvalues_'+str(self.name)+'.pdf')
        plt.savefig(subPath+'qvalues.png', dpi=1500)
     #   plt.savefig(subPath+'qvalues'+str(self.name)+'.svg')
        plt.close(fig)

    return self.decisionThreshold[r,c,0], self.decisionThreshold[r,c,1]


def qvalues_figure2(selves, intersections, figSave):
    # rumination with varying R
    beliefRange = selves[0].beliefRange

    fig,axsr = plt.subplots(1,1)
    fig.set_figheight(6)
    fig.set_figwidth(6)
    # fig.set_figheight(5)
    # fig.set_figwidth(5)
    lines = ["-", "dashed", ":"]
    
    lineColors = ["#FEC400","#FFE035","#FFFC55"]
    A1Colors = ["#5E5EDA", "#ccb3e8", "#ded1f3"]
    A1labels = ['$a_{act}; R_1=-800$', '$a_{act}; R_1=-400$', '$a_{act}; R_1=-200$']
    A3labels = ['$a_{que}; R_1=-800$', '$a_{que}; R_1=-400$', '$a_{que}; R_1=-200$']
    Bintlabels = ['$B=0.18$', '$B=0.29$', '$B=0.4$']
    # plt.rcParams.update({'font.size': 18})
    
    yLims = [-400,110]
            
    for c in range(3):
        axsr.plot([intersections[c],intersections[c]],yLims,'k',linestyle=lines[c],linewidth=3, label=Bintlabels[c])

    axsr.plot([0,1],[0,0],color = 'k', linestyle = '-')
    for s in range(3):
        actionValues = selves[s].actionValues[0,0,:,:]
        # axsr.set_title(self.name)
        axsr.plot(beliefRange,actionValues[:,0],A1Colors[s],linewidth=5, label=A1labels[s])
    axsr.plot(beliefRange,actionValues[:,1],'#DC267F',linewidth=5, label='$a_{r&r}$')

    for c in range(3):
        sampleValues = selves[c].actionValues[0,0,:,2]
        axsr.plot(beliefRange,sampleValues,lineColors[c],linestyle=lines[c],linewidth=5, label=A3labels[c]) #WILL ONLY PLOT ONE IN FIRST 

    # axsr.set(xlim = [0,1],xticks = [0,0.5,1],xlabel ='Belief State $B$')
    axsr.set(xlim = [0,1],xticks = [0,0.5,1])
    axsr.set(ylim = yLims)

    axsr.spines['left'].set_linewidth(3)  
    axsr.spines['bottom'].set_linewidth(3)
    axsr.spines['right'].set_linewidth(3)  
    axsr.spines['top'].set_linewidth(3)  

    # if r == 0:
        # axsr.set(ylabel='Value of Action')
              

    if (figSave != ""): #...print individual to file

        # create the full path for the subfolder
        subPath = os.path.join(figSave, "qvalues/")

        # check if the subfolder already exists
        if not os.path.exists(subPath):
        # create the subfolder if it does not exist
            os.makedirs(subPath)
        
        # plt.savefig(subPath+'qvalues_'+str(self.name)+'.pdf')
        plt.savefig(subPath+'qvalues_figure2.png', dpi=1500)
     #   plt.savefig(subPath+'qvalues'+str(self.name)+'.svg')
        

        # axsr.legend()
        axsr.legend(loc="best")
        label_params = axsr.get_legend_handles_labels() 
        figl, axl = plt.subplots()
        axl.axis(False)
        axl.legend(*label_params, loc="center", bbox_to_anchor=(0.5, 0.5))
        figl.savefig(subPath+'qvalues_figure2_legend.png', dpi=1500)


        plt.close(fig)
        plt.close(figl)        

    return

def qvalues_figure2_punishmentsens(selves, figSave):
    # rumination with varying R
    beliefRange = selves[0].beliefRange

    fig,axsr = plt.subplots(1,1)
    fig.set_figheight(6)
    fig.set_figwidth(6)
    
    A1Colors = ["#5E5EDA", "#9591E1", "#ccb3e8", "#ded1f3"]
    A1labels = ['$a_{act}; R_1=-800$', '$a_{act}; R_1=-600$', '$a_{act}; R_1=-400$', '$a_{act}; R_1=-200$']
    yLims = [-800,100]
        
    axsr.plot([0,1],[0,0],color = 'k', linestyle = '-')
    for s in range(4):
        actionValues = selves[s].actionValues[0,0,:,:]
        # axsr.set_title(self.name)
        axsr.plot(beliefRange,actionValues[:,0], color=A1Colors[s],linewidth=5, label=A1labels[s])
    axsr.plot(beliefRange,actionValues[:,1],'#DC267F',linewidth=5, label='$a_{r&r}$')

    # axsr.set(xlim = [0,1],xticks = [0,0.5,1],xlabel ='Belief State $B$')
    axsr.set(xlim = [0,1],xticks = [0,0.5,1])
    axsr.set(ylim = yLims)

    axsr.spines['left'].set_linewidth(3)  
    axsr.spines['bottom'].set_linewidth(3)
    axsr.spines['right'].set_linewidth(3)  
    axsr.spines['top'].set_linewidth(3)  

    # if r == 0:
        # axsr.set(ylabel='Value of Action')
    axsr.legend()         

    if (figSave != ""): #...print individual to file

        # create the full path for the subfolder
        subPath = os.path.join(figSave, "qvalues/")

        # check if the subfolder already exists
        if not os.path.exists(subPath):
        # create the subfolder if it does not exist
            os.makedirs(subPath)
        
        
        plt.savefig(subPath+'qvalues_figure2_punishmentsens.png', dpi=1500)    
        plt.close(fig)
    return
    
def qvalues_figure3(selves, figSave):
    # deficits in planning 
    beliefRange = selves[0].beliefRange

    fig,axsr = plt.subplots(1,1)
    fig.set_figheight(8)
    fig.set_figwidth(8)
    # fig.set_figheight(5)
    # fig.set_figwidth(5)
    lines = ["-", "dashed", ":"]
    
    # lineColors = ["#FEC400","#FFE035","#FFFC55"]
    lineColors = ["#FEC400"]
    A3labels = ['$a_{que}; k_{max}=5$', '$a_{que}; k_{max}=10$', '$a_{que}; k_{max}=15$']
    # plt.rcParams.update({'font.size': 18})
    
    yLims = [-20,100]
            

    axsr.plot([0,1],[0,0],color = 'k', linestyle = '-')
    actionValues = selves[0].actionValues[0,0,:,:]
    axsr.plot(beliefRange,actionValues[:,0],'#5E5EDA',linewidth=5, label='$a_{act}$')
    axsr.plot(beliefRange,actionValues[:,1],'#DC267F',linewidth=5, label='$a_{r&r}$')

    for c in range(3):
        sampleValues = selves[c].actionValues[0,0,:,2]
        axsr.plot(beliefRange,sampleValues,lineColors[0],linestyle=lines[c],linewidth=5, label=A3labels[c]) #WILL ONLY PLOT ONE IN FIRST 

    # axsr.set(xlim = [0,1],xticks = [0,0.5,1],xlabel ='Belief State $B$')
    axsr.set(xlim = [0,1],xticks = [0,0.5,1])
    axsr.set(ylim = yLims)

    axsr.spines['left'].set_linewidth(3)  
    axsr.spines['bottom'].set_linewidth(3)
    axsr.spines['right'].set_linewidth(3)  
    axsr.spines['top'].set_linewidth(3)  

    axsr.legend()

    # if r == 0:
        # axsr.set(ylabel='Value of Action')
              

    if (figSave != ""): #...print individual to file

        # create the full path for the subfolder
        subPath = os.path.join(figSave, "qvalues/")

        # check if the subfolder already exists
        if not os.path.exists(subPath):
        # create the subfolder if it does not exist
            os.makedirs(subPath)
        
        # plt.savefig(subPath+'qvalues_'+str(self.name)+'.pdf')
        plt.savefig(subPath+'qvalues_figure3.png', dpi=1500)
     #   plt.savefig(subPath+'qvalues'+str(self.name)+'.svg')
        

        # # axsr.legend()
        # axsr.legend(loc="best")
        # label_params = axsr.get_legend_handles_labels() 
        # figl, axl = plt.subplots()
        # axl.axis(False)
        # axl.legend(*label_params, loc="center", bbox_to_anchor=(0.5, 0.5))
        # figl.savefig(subPath+'qvalues_figure2_legend.png', dpi=1500)


        plt.close(fig)
        # plt.close(figl)        

    return

def qvalues_figure3_balance(selves, intersections, figSave):
    # for showing balance between info gain and phasic pain
    beliefRange = selves[0].beliefRange

    fig,axsr = plt.subplots(1,1)
    fig.set_figheight(8)
    fig.set_figwidth(8)
    # fig.set_figheight(5)
    # fig.set_figwidth(5)
    lines = ["-", "dashed", ":"]
    
    # lineColors = ["#FEC400","#FFE035","#FFFC55"]
    lineColors = ["#FEC400"]
    A3labels = ['$a_{que}; r(s=1, a_{que})=-4$', '$a_{que}; r(s=1, a_{que})=-16$']
    # plt.rcParams.update({'font.size': 18})
    
    # yLims = [-20,100]
    yLims = [-75,100]
            

    axsr.plot([0,1],[0,0],color = 'k', linestyle = '-')
    actionValues = selves[0].actionValues[0,0,:,:]
    axsr.plot(beliefRange,actionValues[:,0],'#5E5EDA',linewidth=5, label='$a_{act}$')
    axsr.plot(beliefRange,actionValues[:,1],'#DC267F',linewidth=5, label='$a_{r&r}$')
    
    # axsr.plot([intersections[0],intersections[0]],yLims,'k',linestyle=lines[0],linewidth=3)
    # axsr.plot([intersections[1],intersections[1]],yLims,'k',linestyle=lines[0],linewidth=3)
    # axsr.plot([intersections[2],intersections[2]],yLims,'k',linestyle="--",linewidth=3)
    # axsr.plot([intersections[3],intersections[3]],yLims,'k',linestyle="--",linewidth=3)

   
    sampleValues = selves[0].actionValues[0,0,:,2]
    axsr.plot(beliefRange,sampleValues,lineColors[0],linestyle=lines[0],linewidth=5, label=A3labels[0]) #WILL ONLY PLOT ONE IN FIRST 
    print(sampleValues)

    sampleValues = selves[1].actionValues[0,0,:,2]
    axsr.plot(beliefRange,sampleValues,lineColors[0],linestyle=lines[1],linewidth=5, label=A3labels[1]) #WILL ONLY PLOT ONE IN FIRST 
    print(sampleValues)

    # axsr.set(xlim = [0,1],xticks = [0,0.5,1],xlabel ='Belief State $B$')
    axsr.set(xlim = [0,1],xticks = [0,0.5,1])
    axsr.set(ylim = yLims)

    axsr.spines['left'].set_linewidth(3)  
    axsr.spines['bottom'].set_linewidth(3)
    axsr.spines['right'].set_linewidth(3)  
    axsr.spines['top'].set_linewidth(3)  

    axsr.legend()

    # if r == 0:
        # axsr.set(ylabel='Value of Action')
              

    if (figSave != ""): #...print individual to file

        # create the full path for the subfolder
        subPath = os.path.join(figSave, "qvalues/")

        # check if the subfolder already exists
        if not os.path.exists(subPath):
        # create the subfolder if it does not exist
            os.makedirs(subPath)
        
        # plt.savefig(subPath+'qvalues_'+str(self.name)+'.pdf')
        plt.savefig(subPath+'qvalues_figure3balance.png', dpi=1500)
     #   plt.savefig(subPath+'qvalues'+str(self.name)+'.svg')
        

        # # axsr.legend()
        # axsr.legend(loc="best")
        # label_params = axsr.get_legend_handles_labels() 
        # figl, axl = plt.subplots()
        # axl.axis(False)
        # axl.legend(*label_params, loc="center", bbox_to_anchor=(0.5, 0.5))
        # figl.savefig(subPath+'qvalues_figure2_legend.png', dpi=1500)


        plt.close(fig)
        # plt.close(figl)        

    return


def qvalues_figure4(figSave):

    plt.rcParams.update({'font.size': 30})
    #illus/punishment sensitivity
    # beliefRange = self.beliefRange

    fig,axsr = plt.subplots(1,1)
    fig.set_figheight(9)
    fig.set_figwidth(9)

    lines = ["-", "dashed", ":"]
    

    
    # yLims = [-400,100]
    yLims = [-4,1]
            

    axsr.plot([0,1],[0,0],color = 'k', linestyle = '-')
    # actionValues = self.actionValues[0,0,:,:]
    # axsr.plot(beliefRange,actionValues[:,0],'#5E5EDA',linewidth=5, label='$a_{act}$')
    # axsr.plot(beliefRange,actionValues[:,1],'#DC267F',linewidth=5, label='$a_{r&r}$')

    # axsr.plot(np.linspace(0., 1., 101), np.linspace(100., -400., 101),'#5E5EDA',linewidth=5, label='$a_{act}$')
    # axsr.plot(np.linspace(0., 1., 101), np.linspace(-100., 100., 101),'#DC267F',linewidth=5, label='$a_{r&r}$')
    axsr.plot(np.linspace(0., 1., 101), np.linspace(0., -4., 101),'#FEC400',linewidth=5, label='$a_{que}$')
    axsr.plot(np.linspace(0., 1., 101), np.linspace(-0.5, -0.5, 101),'#40E0D0',linewidth=5, label='$a_{nul}$')


    # axsr.set(xlim = [0,1],xticks = [0,0.5,1],xlabel ='Belief State $B$')
    axsr.set(xlim = [0,1],xticks = [0,0.5,1])
    axsr.xaxis.set_major_formatter(FormatStrFormatter('%.1g'))  # Use '%.1g' for concise formatting
    axsr.set(ylim = yLims)

    axsr.spines['left'].set_linewidth(3)  
    axsr.spines['bottom'].set_linewidth(3)
    axsr.spines['right'].set_linewidth(3)  
    axsr.spines['top'].set_linewidth(3)  

    axsr.legend()

              

    if (figSave != ""): #...print individual to file

        # create the full path for the subfolder
        subPath = os.path.join(figSave, "qvalues/")

        # check if the subfolder already exists
        if not os.path.exists(subPath):
        # create the subfolder if it does not exist
            os.makedirs(subPath)
        
        plt.savefig(subPath+'qvalues_figure4.png', dpi=1500)
        plt.close(fig)
    return

def qvalues_figure5(self,rewardID,costID,figSave):
    # example of belief updating
    beliefRange = self.beliefRange

    fig,axs = plt.subplots(1,len(rewardID))
    fig.set_figheight(8)
    fig.set_figwidth(8*len(rewardID))
    lines = ["--"]
    # lines = [":","dashed","-"]
    lineColors = ["#FEC400","#FEC400","#FEC400"]
    # plt.rcParams.update({'font.size': 18})
    
    yLims = [-20,100]

    for r,reward in enumerate(rewardID):
        
        for c,cost in enumerate(costID): #print the threshold lines
            
            qVals = self.actionValues[reward,cost,:,:]
            
            if len(rewardID)==1:
                axsr = axs
            else:
                axsr = axs[r]

            axsr.plot([self.decisionThreshold[r,c,0],self.decisionThreshold[r,c,0]],yLims,'k',linestyle=lines[c],linewidth=3)
            axsr.plot([self.decisionThreshold[r,c,1],self.decisionThreshold[r,c,1]],yLims,'k',linestyle=lines[c],linewidth=3)      
        
        actionValues = self.actionValues[reward,0,:,:]
        # axsr.set_title(self.name)
        axsr.plot([0,1],[0,0],color = 'k', linestyle = '-')
        axsr.plot(beliefRange,actionValues[:,0],'#5E5EDA',linewidth=5, label='$a_{act}$')
        axsr.plot(beliefRange,actionValues[:,1],'#DC267F',linewidth=5, label='$a_{r&r}$')

        

        for c,cost in enumerate(costID):  #then print the three costs

            sampleValues = self.actionValues[reward,cost,:,2]
            axsr.plot(beliefRange,sampleValues,lineColors[c],linestyle="-",linewidth=5, label='$a_{que}$') #WILL ONLY PLOT ONE IN FIRST 


        n_samp = np.int32(self.noSamples_t[0,0,0])
        print(self.posterior_beliefs[0,:n_samp+1])
        for n in range(n_samp+1):
            axsr.plot(self.posterior_beliefs[0,n], 0, "xr", markersize=15, markeredgewidth=5)

        for b in [0.5, 0.84, 0.92]:
            axsr.plot(b, 0, "xb", markersize=15, markeredgewidth=5)

        # axsr.set(xlim = [0,1],xticks = [0,0.5,1],xlabel ='Belief State $B$')
        axsr.set(xlim = [0,1],xticks = [0,0.5,1])
        axsr.set(ylim = yLims)

        axsr.spines['left'].set_linewidth(3)  
        axsr.spines['bottom'].set_linewidth(3)
        axsr.spines['right'].set_linewidth(3)  
        axsr.spines['top'].set_linewidth(3)  

        # axsr.legend(loc='lower left')

        # if r == 0:
            # axsr.set(ylabel='Value of Action')
              

    if (figSave != ""): #...print individual to file

        # create the full path for the subfolder
        subPath = os.path.join(figSave, "qvalues/")

        # check if the subfolder already exists
        if not os.path.exists(subPath):
        # create the subfolder if it does not exist
            os.makedirs(subPath)

        # plt.savefig(subPath+'qvalues_'+str(self.name)+'.pdf')
        plt.savefig(subPath+'qvalues_figure5.png', dpi=1500)
     #   plt.savefig(subPath+'qvalues'+str(self.name)+'.svg')
        plt.close(fig)

    return self.decisionThreshold[r,c,0], self.decisionThreshold[r,c,1]
