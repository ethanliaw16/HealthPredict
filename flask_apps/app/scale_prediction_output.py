import numpy as np

def scale_output(outcome, gbm_scalers):
    pred_min = gbm_scalers[0]
    pred_threshold = gbm_scalers[1]
    pred_max = gbm_scalers[2]
    scaled_outcome = outcome

    if(outcome > pred_max):
        outcome = pred_max
    if(outcome < pred_min):
        outcome = pred_min
    
    if(outcome >= pred_threshold):
        scaled_outcome = ((outcome - pred_threshold)/(2 *(pred_max - pred_threshold))) + .5
    else:
        scaled_outcome = ((outcome - pred_min) / (2 * (pred_threshold - pred_min)))
    
    return scaled_outcome