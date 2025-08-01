# Injury POMDP
Code accompanying manuscript - Homeostasis after injury: how intertwined inference and control underpin post-injury pain and behaviour

For a very basic demo/tutorial, please use this Colab notebook: https://colab.research.google.com/drive/1bDWlrGguXCrJG-GaIXkOEfbuucqcIsnF?usp=sharing

## Overview
`normative_injury_investigation_and_trade-off.py` generates the results for the normative consequences section of the paper. `wrong_environment_info_restriction.py`, `wrong_problem_overestimating_injury.py` and `wrong_problem_underestimating_injury.py` generate results for the dysfunctional consequences section of the paper. 

For simulations without $a_{nul}$ action, `build_object_rumPOMDP.py` builds the POMDP and runs belief grid value iteration.`qvalues_rumPOMDP.py` is used to generate Q-vlaue plots and `sample_sequence_functions_rumPOMDP.py` is used to sample a sequence of observations and take actions to get belief trajectories. For simulations including $a_{nul}$, `build_object_rumPOMDP_twocostlyactions.py`, `qvalues_rumPOMDP_twocostlyactions.py` and `sample_sequence_functions_rumPOMDP_twocostlyactions.py` does the same. Rest of the files include helper functions.

The requirements are numpy, scipy, matplotlib and pandas.

Contact pranav.mahajan AT ndcn.ox.ac.uk for further details.
