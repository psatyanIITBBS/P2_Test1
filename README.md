# **Project 02 - Self Driving Car Nano Degree** 

## Writeup

---

**Goals of the Project**

The goals / steps of this project are:

* To load the data set 
* To explore, summarize and visualize the data set
* To design, train and test a model architecture
* To use the model to make predictions on new images
* To analyze the softmax probabilities of the new images
* To summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Distribution.png "Distribution"
[image2]: ./examples/CheckItem.png "CheckItem"
[image3]: ./examples/Five.png "Five"
[image4]: ./examples/CLAHE_effect.png "CLAHE"
[image5]: ./examples/CLAHE_Gray_effect.png "CLAHE_Gray"
[image6]: ./examples/Five_new.png "FiveNew"


## Rubric Points
### Here, following the provided template, I have considered the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and described how I have tried to address each point in my implementation.  

---
### Load the Data Set
#### 1. Download and load the data set
The data set was downloaded from the link:
https://s3.amazonaws.com/video.udacity-data.com/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip

for getting the data to the AWS server the following steps were used:

**Step 1:** Got to the directory where the Jupyter notebook was kept.

**Step 2:** wget https://s3.amazonaws.com/video.udacity-data.com/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip

**Step 3:** unzip traffic-signs-data.zip

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python, numpy and pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32x32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

First I created a function called "SplitClass" to split any data set into its classes so that it will be easier to investigate any particular class of images if and when required.

For an exploratory visualization of the distribution of various kindls of signs in the training data set, the following bar chart was prepared to visualize the distribution of all the classes from 0 through 42.

![alt text][image1]

Using the "SplitClasses" function and the pandas library we can plot any item of any desired class. For example, by choosing the class number to be 0 and item number to be 111 we get the index of the item in the main training data and with this index and the "signnames.csv" file we can plot the following figure.

![alt text][image2]

Then another function was written to choose and plot five random images from any given data set. The output of one such call to this function produces something like this:

![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

As a first step, I decided to enhance the contrast of the images through the application of CLAHE filter as Contrast Limited AHE (CLAHE) is a variant of adaptive histogram equalization in which the contrast amplification is limited, so as to reduce the problem of noise amplification. Here is an example of a traffic sign image before and after applying CLAHE filtering.

![alt text][image4]

In the next step of improvement, I changed the images into grayscale data. The images after passing through this filter is shown below:

![alt text][image5]

The difference between the original data set and the processed data set can be clearly seen in the above example. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, 'Valid' padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, 'Valid' padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten	    | output 400x1      									|
| Fully connected		| input 400, output 120        									|
| RELU					|												|
| Fully connected		| input 120, output 84        									|
| RELU					|												|
| Fully connected		| input 84, output 43        									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I started with small batch sizes (from 4 upto 512 in every step doubling the size). The rate was kept at 0.001 for most of the training. However, sometimes, for larger batch sizes, the rate was reduced to even 0.0001. everytime the model got a better accuracy, the weights were saved and the new weights were used in the next epoch. In every battch size about 40 epochs were used on an average.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results arere:

* training set accuracy of 99.8%
* validation set accuracy of 94% 
* test set accuracy of 91.2%


If a well known architecture was chosen:
* Finally the LeNet architecture only happened to give the best accuracy.
* Though the accuracy level on the test data was just above 91%, still it is able to predict most of the new signs quite accurately. However, the model can be improved a lot in many aspects.
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6]

The first image might be difficult to classify because the image is too bright. The second image may be difficult because it is too dark and the angle of view is also tilted. The third on may be difficult because of the perspective. The fourth one is difficult because of the motion blur and the fifth one may be difficult because of the low light condition.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction from the model:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Turn right ahead     			| Turn right ahead 										|
| Road work				| Road work											|
| Wild animals crossing	      		| Wild animals crossing					 				|
| Yield			| Yield      							|


The model was able to correctly guess all the 5 traffic signs, which gives an accuracy of 100%. This is really surprisin in view of the accuracy on the test set being merely 91.2%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the all the images, the model is found to be quite sure that actual sign is the predicted sign (probability of almost 1.0). Actually, the other probabilities are so low that they should not even be mentioned. May be the images were quite obvious for the model to predict. The below table, for the above mentioned reasons, shows that in all the five cases the Softmax Probability of the best prediction almost reached 100%.

| Probability of prediction         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100%         			| Stop sign   									| 
| 100%     				| Turn right ahead 										|
| 100%					| Road work											|
| 100%	      			| Wild animals crossing					 				|
| 100%				    | Yield      							|


### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! 
