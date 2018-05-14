# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./recovery_driving_fromright.JPG "Image1"
[image2]: ./recovery_driving_fromleft.JPG "Image2"
[image3]: ./center_driving_clockwise.JPG "Image3"
[image4]: ./center_driving_anticlockwise.JPG "Image4"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality


#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```


#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.



### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of the following layers:
    1. 4 5x5 convolution layers with a stride of 1x1 and different outputs: 6, 16, 32, 64 and ReLu as activation function (to add non-            linearity to the model).
    2. 3 Max Pooling layers with a filter size and stride of 2x2.
    3. 5 FC layers with output sizes of 512, 256, 128, 64, 1.

Also the input data has been preprocessed before feeding into the model to make sure an evenly feature distribution. Steps contain:
    1. Gray scaling of the input images (As this has been done outside the Keras model architecture using opencv, same step added to              drive.py also during autonomous driving mode.
    2. Normalization of the image data.
    3. Cropped the input images (as suggested in the lectures to reduce any noise from the data - 70 pixels from the top, 20 from the              bottom).


#### 2. Attempts to reduce overfitting in the model

I collected the training data by driving the car for couple of laps in both the directions (so that the model can generalize better and doesn't overfit). After training my model on the data I collected (drove the car various times with center lane driving, recovery from left and right edges wherever needed), managded to keep the car on track so didn't feel the need of adding dropout layers in the model.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

My final model used an adam optimizer, so the learning rate was not tuned manually just used the default value (model.py line 91). Although I tried with other optimizers like adagrad, adamax but those turned out to perform too bad because I observed very high training and validation loss and training was taking too many epochs to reduce the loss. So switched back to adam optimizer.

I also tuned the batch size used in generator() function and finally used a size of 64.


#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Here is the approach I took:

   1. Initially collected training data using center lane driving one lap in both the directions, trained my model on the data for 5-6           epochs (stopped after 5-6 epochs as accuracy didn't improve much further or model was overfitting if trained for more epochs) and           quickly tested on the simulator.
   2. Drove around one more lap again in both directions to get more data as car was unable to stay on track at various occasions,               retrained the model and tested on the simulator.
   3. After making changes to model architecture and training it for multiple attempts, tested the model on simulator and got impressive         results (car was able to drive on track, although not at the turns).
   4. Further collected some data by doing recovery driving multiple times from left and right directions (based on the performance of the         car at turns and wherever needed) so that car can learn to take turns in left and right directions.
   5. I didn't flip the input images as I already captured lot of data by driving in both directions around the lap.



### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was as follows:

   1. My first step was to start with some base model architecture for training on the data (as mentioned and explained in the lectures).         Without spending much time looking around, I thought of starting with the Traffic Signal classifier model I submittted in the               previous project and added another convolution layer and couple of FC layers (as the input image size I had here was 160x320 as             compared to 32x32).
   2. Quickly collected some training data using center lane driving, added preprocessing steps (as discussed above) and trained the model       for 5-6 epochs with batch size of 32 using Adam optimizer by splitting the data into train and validation sets (80-20).
   3. Inititally the mean squared error was high on both validation and train sets as I didn't collect much data, so drove for another lap       in both directions to collect more data.
   4. After multiple attempts of training the model and testing on simulator, car was able to drive on track quite well although it was           unable to judge well on the turns. Also tweaked the model architecture few times (adding and removing conv layers and changing the         number of outputs to finally reach at the final architecture).

Now the problem was to deal with driving at turns, then as suggested in the lectures about data collection tactics I started doing recovery driving (drive from the edge towards the center and collect the data). This is now when it became interesting and tricky, here is how I proceeded further:

   1. Collected lot of data step by step by driving from left towards center, from right towards center at different turning angles and           positions, started and stopped recording etc.
   2. Trained the model again along with new data, when tested the model on simulator the car started driving really great on the turns           too.
   3. Performed few more rounds of collecting data, finally car was able to drive autonomously staying on the track for the whole time.           Thats when I felt so great and happy that finally I was able to get throught !! :), although it happened over the course of many days       and it took me lot of time to reach this stage.
   4. Still some improvement was need, as the car was still drving near the right edge for some part of the track. But since now I had a         saved model with pre-trained weights that could drive car on the track, I thought why to now train the model from scratch instead I         should use the pre-trained model to improve further on a subset of data. eg. I collected a little training data driving from right         edge towards center at multiple positions and angles (also flipped the images to generalize well so that car is able to stay on track       if going in wrong direction), then trained the existing model on this data to improve further.
   5. Finally I got some improvement and car started driving in the center but not everywhere. Beyond this point, I couldn't improve             further even by collecting data again and again.

In my submission, I have submitted 2 versions of video.py - initial and final, showing the transition of improvement in driving.


NOTE - I was only using the center images as input from training data and steering angles as the label y.


#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of the follwing layers:

   1. 4 5x5 convolution layers with a stride of 1x1 and different outputs: 6, 16, 32, 64 and ReLu as activation function (to add non-             linearity to the model).
   2. 3 Max Pooling layers with a filter size and stride of 2x2.
   3. 5 FC layers with output sizes of 512, 256, 128, 64, 1.


#### 3. Creation of the Training Set & Training Process

I have already discussed above how I collected the training data, just displaying few images below from different driving behaviours. 

Images from center lane driving:

![alt text][image3]
![alt text][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image1]
![alt text][image2]


