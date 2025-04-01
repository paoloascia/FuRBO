# FuRBO Stopping criterion
# 
# March 2024
##########
# Imports

def failed_GP(state):
    if state.failed_GP:
        # print('Restart')
        return True
    return False
    