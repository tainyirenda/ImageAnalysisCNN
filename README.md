<h1 align="center">Hi 👋, I'm Tai</h1>
<h3 align="center"> Welcome to my Image Analysis Project.</h3>
<br/>
<h4>  🌱 Check out how I used a Deep Learning solution with a Convolutional Neural Network (CNN) and Data Augmentation techniques for Image Analysis. </h4>

# Image Analysis Using CNN (Covolutional Neural Network)

**Introduction:** <br/>
This report focuses on building a Convolutional Neural Network (CNN) to classify images of cats and dogs. The dataset contains 25,000 images, with 12,500 images in each class. Due to the large volume of data, a CNN model was deemed most appropriate, given its capacity to handle large datasets, unlike other models such as Artificial Neural Networks (ANNs) that struggle with numerous parameters in training data. CNN models have been widely used in computer vision since 2012 as part of Deep Neural Network Architecture, outperforming other techniques in classification (Krizhevsky et al., 2012).
In addition to the CNN model used in this report, data augmentation techniques have been applied to improve object recognition, enhance model performance, and prevent overfitting. In addition, having a large dataset for training allows the model to better learn and recognize objects, considering the variability objects can exhibit in real-world settings.
Packages
Packages used throughout were as follows:
•	NumPy (for dataset arrays)
•	Matplotlib (for plotting)
•	TensorFlow (with tfds dataset management)
•	PIL (for image augmentation)
•	Keras (for image preprocessing and model layers)
•	SciKit Learn (train/test split of dataset for model selection)

**The Model:** <br/>
The model consists of convolutional and pooling layers for feature extraction, followed by fully connected layers for classification. The Adam optimizer is used for training, and binary cross entropy is the loss function, which is suitable for binary classification problems. The model aimed to minimize this loss during training. The accuracy metric was used to evaluate the performance of the model during training and testing. The layers and parameters are as follows:

•	Input Layer:
Type: Conv2D
Filters: 32
Kernel Size: (3, 3)
Activation Function: ReLU
Input Shape: (224, 224, 3) - Assumes input images are RGB with size 224x224 pixels.

•	Max Pooling Layer:
Type: MaxPooling2D
Pool Size: (2, 2)
This layer reduces the spatial dimensions of the representation.


•	Convolutional Layer:
Type: Conv2D 
Filters: 64
Kernel Size: (3, 3)
Activation Function: ReLU

•	Max Pooling Layer:
Type: MaxPooling2D
Pool Size: (2, 2)

•	Flatten Layer:
This layer flattens the 2D output to a 1D array before feeding it into the fully connected layers.

•	Dense (Fully Connected Layer):
Type: Dense
Units: 128
Activation Function: ReLU

•	Output Layer:
Type: Dense
Units: 1
Activation Function: Sigmoid
This is a binary classification task, so the sigmoid activation is used to output values between 0 and 1, representing the probability of belonging to class 1.

•	Compilation:
Optimizer: Adam
Loss Function: Binary Cross entropy
Metrics: Accuracy

**The Dataset/Data Pipeline:** <br/>
The dataset is comprised of a parent folder (named ‘Pet Images’) which then has subdirectory folders, one named ‘Cat’ and one named ‘Dog’. These subdirectories are where the images of each class of Cat and Dog are stored with variability in image size. During the extraction of data, some images appeared to be corrupt and were removed. The size of the dataset obtained and used for training then reduced to 24972 images in total belonging to the two classes. The data was loaded in using Image Generator from the  
‘tensorflow.keras.preprocessing.image’ module, providing the data generator instance with parameters of an image rescale to normalise pixel values of the images to values in the range of 0 to 255, and a valid split of 0.2 (20%). The images were given a batch size parameter of 32 of which would be iterated through and a target size parameter of 244x244 to resize all images to the same size that fits the model parameters for the first Conv2D sequential layer of the model.

Once loaded, images were converted into NumPy arrays. This is because the PIL package used to apply augmentation techniques to the dataset only works with data in array format. Once augmented, images were reverted from arrays for the model. A sample image was extracted, as shown in figure 1 below, from the dataset to test application techniques for the images preprocessing with conversion to RGB colour scale. The dataset was split into train and test subsets with a 20/80 split (20% validation/80% training). Only the first 100 images of the training dataset batch were pre-processed with augmentation techniques before being passed to the model, with the dataset batch iterated every time.
