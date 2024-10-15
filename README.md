# DNN-Initialised-Framework
Framework where DNN predicts the initial flow field, which is then used to initialise the CFD solver and later provides aerodynamic coefficients.

In the DNN-Initialised CFD method, the flow field predicted by the DNN is used to generate an initialisation file for the CFD solver. Once converged, the aerodynamic coefficients (C_L and C_D) are calculated from the final CFD solution.   

First, aerofoil coordinates are used to generate the SDF array, which, along with flow conditions, is fed into the DNN model to predict the aerofoil's flow field. The field is then denormalised using the original global minimum and maximum values of the flow field variables from the training dataset. This denormalised flow field is mapped onto the mesh around the aerofoil to generate the initialisation file. Values are assigned by nearest point interpolation, with non-interpolated points on the edges filled accordingly, and any remaining points are assigned a fill value of 1. Finally, the file is used in the FLITE2D CFD solver to compute the final aerofoil flow field.  

The SDF generator and DNN UI are available in repository titled "DNN".  
Link: https://github.com/Asif-MushtaqAA/DNN  

FLITE2DonPY is available in repository titled "FLITE2D-on-Py".  
Link: https://github.com/Asif-MushtaqAA/FLITE2D-on-Py  

Note: Make sure to use correct paths for added repositories.

Example Implementation in console  
from DNN_FLITEonPY import ResidualBlock, ChannelSpecificDecoder, EncoderDecoderCNN, workflow  
workflow(58,0.6,2)  
