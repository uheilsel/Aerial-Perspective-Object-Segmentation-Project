# SEP740

## Description:
Drone and aerial picture-taking quality has improved drastically in the past decade. Drone stabilization allows pictures taken from an aerial view to be crystal clear without shaking or blurriness. This has many practical and exciting applications for photography, cinematography, and also image recognition! Drone images can be used to quickly identify people and seek out specific objects in a large area. Think of how this could be used for spotting survivor rescues in disaster-struck areas.

The Semantic Drone Dataset focuses on semantic understanding of urban scenes for increasing the safety of autonomous drone flight and landing procedures. The imagery depicts more than 20 houses from nadir (bird's eye) view acquired at an altitude of 5 to 30 meters above the ground. A high-resolution camera was used to obtain images at a size of 6000x4000px (24Mpx).

The dataset consists of 400 images that have been annotated according to twenty standard classes such as trees, persons, cars, and pavement.

See more information at the dataset website.

## Project Outcomes:
Identify everyday objects such as cars and roads in bird's eye view images 

Use a trained model to identify objects over a large, continuous, mapped area (i.e., your local neighborhood from google maps)

Use the positioning of cars and people determined to flag areas where pedestrians may be at most risk of an accident

## Report 1:
Using the iris dataset (https://archive.ics.uci.edu/ml/datasets/iris), I designed and developed a machine learning model for the algorithms given below and predicted the outcome. Then, I compared the results of F1 score, CM, Accuracy and timings for various algorithms. Algorithms are:

K-Nearest Neighbours (KNN). 
Gaussian Naive Bayes (NB).
Support Vector Machines (SVM).

My code include Visualize the dataset and its class distribution; Performing cross-validation; Comparing various model wrt Confusion Matrix, Accuracy and F1 score; And calculating timing for training and evaluation for each algorithm and compare the timings.

## report 2:
I and my teammates designed and developed a classification model using CIFAR-10 photo dataset in Keras with visualizing the data along with classes. Followed we pre-processed our data-->Drop/correct classes and normalize. After tat, we designed convolution neural network and our choice for the hyper parameters. Then we evaluated the model and identified does it have over fitting or under fitting problems. Finally, we applied hyper parameter tunning using Grid search.
