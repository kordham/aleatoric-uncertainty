# aleatoric-uncertainty
Predicting the heteroscedastic aleatoric uncertainty of the images of radio galaxies


## Abstract
A convolutional neural network was trained to  predict the heteroscedastic aleatoric uncertainties of images as a learned parameter.  This was done by introducing a normal prior distribution over the output layer whose standard deviation was derived from a secondary output of the network.  This network was then trained on a data set including 50x50 images of radio galaxies, and by the end of training it predicted the aleatoric uncertainties of input images.  
 This was tested by augmenting the test data and measuring the change in the distribution of uncertainty pre- and post-augmentation. Then, the training data set was split into five quintiles based on uncertainty and separate models were trained on them. The models were compared with each other as well as with a model trained on randomly pruned data by using their validation accuracy and loss curves.  The second lowest uncertainty quintile had the highest validation accuracy and the lowest validation loss, outperforming the rest of the models. This was identified to be due to its spread of data between low uncertainty data and some uncertain data where more new features are displayed. 
