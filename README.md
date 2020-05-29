# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/trainset_examples.png "Training set"
[image2]: ./output_images/validationset_examples.png "Validation set"
[image3]: ./output_images/testset_examples.png "Test set"
[image4]: ./output_images/trainset_preprocessed.png "Training set preprocessed"
[image5]: ./output_images/web_images_preprocessedd.png "Web image set preprocessed"
[image6]: ./output_images/web_images.png "Web image set"
[image7]: ./output_images/web_images_predictions.png "Web images predictions"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

To visualize the dataset, I utilized the subplots function to plot 6 images from each set.
Training set:

![alt text][image1]

Validation set:

![alt text][image2]

Test set:

![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to use grayscaling as an initial step in order to remove color biasing in my neural network.
The next step was to use rank equalized histogram to improve the images with low contrast.
Lastly, images were normalized to obtain a standardized dataset. Each image was normalized to a scale of (0,1).

An example of preprocessed training set and the final web_images used are shown below:
Training set:

![alt text][image4]

Web images:

1[alt text][image5]

The training set images were shuffled before being fed to the neural network to remove index based biasing from the learning dataset.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I made use of the standard LeNet-5 network which consists of 2 convolution layers and 3 fully connected layers.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale 							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout				| keep_dims = 1									|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Dropout				| keep_dims = 1									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten		        | output 400  									|
| Fully connected		| output 120   									|
| RELU					|												|
| Dropout				| keep_dims = 1									|
| Fully connected		| output 84   									|
| RELU					|												|
| Dropout				| keep_dims = 1									|
| Fully connected		| logits output 43								|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the AdamOptimizer as compared to the simple gradient descent optimizer, it provides better performance and is commonly relied on for computer vision problems. 
I experimented with a few batch sizes and found that smaller the batch size, better was the accuracy of the predictions but at the cost of computing speed. I eventually settled on a batch size of 50.
For number of epochs, as can be seen from the simulation, the accuracies settled in after about 15. I chose 30 as there were still some fluctuations with each increasing run.

I chose to keep dropout rate as a constant and not a tunable hyperparameter as during my experiments, I found it to perform best when it was around 1 and any decrease in value only hampered the accuracy.

Learn rate was chosen as 0.001 after a few trials. Increasing the run rate made the predictions unreliable as expected due to large movements of the prediction zone.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.8%
* validation set accuracy of 93.3%
* test set accuracy of 91.8%

If a well known architecture was chosen:
* What architecture was chosen? I chose the LeNet-5 architecture and included several layers of maxpooling after each convolution. The addition of maxpooling helped to downsample from the previous layers and bring to focus the parts of the image with high features for extraction. 
All layers except for the final one, included a relu activation and dropout layer which helps in randomizing the prediction capability and reducing the reliance of the neural network on a few features in a particular order. This addition helped in improving the accuracy of the datasets.

* Why did you believe it would be relevant to the traffic sign application? This architecture was introduced as a starting point. So, I built on this by adding a few dropout and relu layers to make it more robust.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 The >90% accuracies for all three datasets shows that the model is performing very well.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6]

I expected the speed signs especially the 70 km/hr sign to get misclassified as 20 or 30, but the neural network had learnt enough to give accurate predictions.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image7]

As shown, the predictions were done with a probability of more than 90% for the correct label among the top 5 choices for each image.
The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

As shown in the image above, each of the images were classified with a probability greater than 90% for the correct label of the image.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


