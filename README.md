# ecg-project
This is Sanda Honor Thesis for JMU CS and ENGR

This research is to explore the performace of machine learning model on an embedded system. The machine learning model is able to classify if there is a heart disease. Coronary artery disease (CAD) is the heart disease that is covered in the scope of this research. There are 4 different ECG waves that can present in CAD. (1) ST-T wave changes, (2) T wave changes, (3) Atrial fibrillation (AF)  and (4) Myocardial infarction (MI). 

At this moment, I will focus on the machine learning model that can classify AF waves. The model that will be used to implement is described in this paper (https://arxiv.org/pdf/1710.06122.pdf). The model can classify between 4 labels: normal, AF, other and noise. 

The embedded system that will used is ESP32 development S3 board.

___
## Goal 

The goal of this research is to implement a machine learning model in a embedded system using tensorflow framework that can perform classification on device in real-time. The next step is to collect the data to compare the performance between the training computer and embedded system to classify the testing data. 

The standard benchmark will be the performance of the model specified in ths paper (https://arxiv.org/pdf/1710.06122.pdf). 

___
## Hypothesis 

In this research, I will experiment if the following hypothesis is true. 
The current model(M) is very big to implement on the embedded system. (Full model takes 14MB and available memory is 8MB)
I hypothesize there exists a model (M') that is smaller than (M) to fit on the embedded system with an acceptable accuracy (>70%). I assume that as the model size gets bigger, the accuracy goes up as well as the prediction time. 



___
## Progress

| Date | Description |
|------|------------ |
|2022 Aug| Ordered the board, download the code base, prepare the development environment|
|2022 Sept 1st week| Found the original research github repo, download and train the model, downgrade the tensorflow to 2.5.0, upgrade the original code to tf v2|
|2022 Sept 2nd week| Struggling with converting to tensorflow v2 on original model, learning on how to use platformIO to program the board, set up environment on levinthal|
|2022 Sept 3rd week| Original TF model is trained on levinthal, acheived similar accuracy (79%), cannot transform to save a model to tflite, finding a way to save the model|
|2022 Sept 4th week| Training the example model from tf-micro (number prediction) on esp32, learning with tflite conversion process
|2022 Oct 1st week| Found an another example tflite model on github that also guide us to use tflite cpp, the git repo contain all code/tflite cpp library to use in the board, implemented on the board and find out the result, model implementation success|
|2022 Oct 2nd week| Found out I cannot use original tf model, because it used low level tensorflow operation (not keras), I don't know how to convert that model to tflite, Dr. Molloy suggested to try on keras model |
|2022 Oct 3rd week| Dr. Forsyth asked me to see if we can actually use CNN on esp32, Tested with very simple tomato cnn model, that was abonded due to long time for training, play around with NN model, had trouble with github and migrate all codes to new repo "ecg_project_2"|
|2022 Oct 4th week| implementing mnist NN on esp32, it compiles, having trouble with sending data through Serial line, do not know how to send data for array.
|2022 Nov 1st week| use 2d array as header file, test on NN mnist model, it works, test on CNN mnist model, best model too big to fit in ram and allocate tensor, use smaller model, it works. |
|2022 Nov 2nd week| Had a meeting with Dr. Molloy and Dr. Forsyth, having a lot of trouble with conversion of tflite, roll back to back up plan, some person in github implemented the same model in keras model, I trained the keras model on levinthal, Dr. Molloy, Dr. Forsyth approved, achieved 85% accuracy, higher than original paper. Prepare the presentation for honor committee. get feedbacks from them to start the experimental model, implementing the keras model on esp32, sending the data through serial, test out other ML algorithms, getting back the result. original keras model takes 14MB, no space on board|
| 2022 Nov 3rd week (Thanksgiving week)| We order new board with 8mb ram, I prepare the data with labels from ptb_xl, train a lot of mini CNN model and NN model with new data and implement on the board. All success except I do not know how to send data in real_time through serial port. |
| 2022 Nov 4th week | I got off-track, Dr. Forsyth push me to stop using new data and focus on using the existing model, he asked me to tweak the existing model. He helped me with sending the series of data. I got the new code from him, tested with sending tiny bit of data. it was success, Downside, it takes 4 mins to send 780 data points, over 30 mins to send 18810 data points. |
|2022 Dec 1st week| I found out my current tflite library is out of date and do not have functions I need, upgrade the tflite library to latest (as of Dec 2022), still missing one function I need, modify the cnn keras model to see if I removed the model, Reevaluating the hypothesis to design the experimental model. Working with Dr. Molloy to evaluate the hypothesis. |
|2022 Dec 2nd week | 




