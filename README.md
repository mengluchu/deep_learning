# Study of deep learning in air pollution

Exploration of air pollution mapping and others. The very initial motivation is to learn the road-no2 relationships automatically than using buffers

* data_arrange.py: gather data, the original data are pcrastermaps of roads in 25 m buffer at coordinates of ground stations
* cnnap: explore a  simple deep cnn structure with augumentation, different batchsize, structure, activation, pooling strateg.

#### current findings 


* I obtained higer accuracy with dropout and batchnorm. The lowest mae is around 10. 
* averagepooling is not as steady as maxpooling
* batch size setting to 100 obtained better results than 32 and 50
* seems to converge at around 5 epochs
* The activation of the last layer is set to relu to indicate no negetive predictions and for better backprop, seems to perform better
* With imagegenerator seems to obtain a more steady result

#### notes
Currently only using 2000 stations for trainging and 600 for testing
data augumentation assumes the inputs are images, with 1, 3, or 4 channels: grayscale, rgb, 
 
