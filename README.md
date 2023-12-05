<h1 align="center">Hi 👋, I'm Tai</h1>
<h3 align="center"> Welcome to my Image Analysis Project.</h3>
<br/>
<h4>  🌱 Check out how I used a Deep Learning solution with a Convolutional Neural Network (CNN) and Data Augmentation techniques for Image Analysis. </h4>

# Image Analysis Using CNN (Covolutional Neural Network)

**Introduction:** <br/>
This project focuses on building a Convolutional Neural Network (CNN) to classify images of cats and dogs. The dataset contains 25,000 images, with 12,500 images in each class. Due to the large volume of data, a CNN model was deemed most appropriate, given its capacity to handle large datasets, unlike other models such as Artificial Neural Networks (ANNs) that struggle with numerous parameters in training data. CNN models have been widely used in computer vision since 2012 as part of Deep Neural Network Architecture, outperforming other techniques in classification (Krizhevsky et al., 2012).
In addition to the CNN model used in this report, data augmentation techniques have been applied to improve object recognition, enhance model performance, and prevent overfitting. In addition, having a large dataset for training allows the model to better learn and recognize objects, considering the variability objects can exhibit in real-world settings.

**Packages:** <br/>
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

**Methods:**<br/>
Augmentation Techniques:<br/>
Random Crop:<br/>
The random crop transformation chooses a random zoom size of a bounding box of the origianl image and crops it at that zoom size. The tensorflow package comes with an API already prefixed for random cropping. With this resizing was also added for the input image to meet the requirements of the first sequential Conv2D model layer.

Random Central Crop:<br/>
Unfortunately, using random crop alone can sometimes produce images with irrelevant sections of the original image, causing the model to classify incorrectly if excessively zoomed and cropped. Due to this, Random central crop was implemented, which uses a ‘central fraction’ argument that crops from the centre of the original image based on its central fraction. With this resizing was also added for the input image to meet the requirements of the first sequential Conv2D model layer.

Random Brightness: <br/>
For random brightness, the ‘ImageEnhance.Brightness’ object from the PIL package was used. This works by increasing the images brightness by a factor randomly chosen from the range 1- max_delta to 1+max_delta. Note that the random brightness does not affect image sizing.

Random Contrast:<br/>
Like the random brightness function, the random contrast functions create an ‘ImageEnhance.Contrast’ object from the PIL package that enhances the original image from a specified range of lower to upper. Note that the random contrast does not affect image sizing.

Random Jitter: <br/>
Random jitter function included a variation of previous techniques to mix and match, and functionality that flips the original image left to right. This was mixed with other methods in a single function as the ‘flip_left_right’ method alone can decrease model accuracy. Additionally, it is more convenient for the model to have the transformations in a single API.

**Results/Evaluation:**<br/>
The CNN classifier model was run with normalization and batch sizing for the training data subset, both with and without augmentation. The model was trained using the ‘.fit()’ method which had the dataset and preprocessing parameters for augmentation over 10 epochs specified and dataset over 10 epochs specified for data without augmentation. 

Results Comparing Model with and without Augmentation: <br/>
Augmentation boosted the model's accuracy by about 0.1 (10%). The augmented model had a lower final loss and the model improved across 10 epochs, while the non-augmented started to plateau after 8 epochs.

**Conclusion:**<br/>
For the CNN model run without any augmentation, the model achieved a final accuracy of 88.07% after 10 epochs. The loss decreased steadily over the epochs, reaching 0.3433 in the final epoch. This indicates that the model without augmentation performed well, demonstrating a steady improvement in accuracy and a significant reduction in loss. The accuracy suggests that the model effectively learned the patterns in the training data. However, the accuracy/loss plateaued quickly compared to the augmented model (at 8 epochs), suggesting limited learning thereafter.

For the CNN model run with augmentation applied, the augmented model outperformed the model without augmentation by a large margin, achieving a higher final accuracy of 98%. This suggests that the model generalizes well and responds effectively to unseen data with augmentation. Augmentation enhances the model's ability to generalize by exposing it to a variety of augmented images during training, resulting in improved performance on unseen data. While the model without augmentation is good, it might be more prone to overfitting, as evidenced by the lower accuracy compared to the model with augmentation. Augmentation is a powerful technique that improves the model's ability to generalize and perform well on new, unseen data.
While data augmentation is generally beneficial for training robust models, there are scenarios where certain augmentation techniques might not improve or could potentially decrease model accuracy. On the other hand, augmentations simulate real-world conditions, helping the model be more adaptable to challenges it may face. Providing multiple techniques allows the model to be trained on a wider variety of data, resulting in a more diversified dataset for better generalization to unseen data.

**Recommendations:**<br/>
The model with augmentation is recommended for its higher accuracy and better generalization, especially if the dataset is limited. However, this can also depend on the dataset, model architecture, and the augmentation techniques applied. For future recommendations, it’s always good practice to validate conclusions through multiple experiments and datasets.
If good model performance is not achieved with other data, methods such as transfer learning can be considered, where the structure of the model is borrowed for use, and the general extraction layer is added as a convolution layer (Simonyan et al, 2014). Deeper neural network depth can significantly improve model performance (Kim et al, 2016). Furthermore, a model checkpoint can be used, where the weights of the trained model at its best accuracy can be saved. Moreover, saving the best model performance as a fine-tuning method can enhance overall performance.
Similarly, ensemble learning, where predictions from multiple models are combined, can often lead to improved performance and more robust results. Different augmentation techniques can affect model performance differently; considerations could include trying other techniques such as shear combined with zoom, contrast, brightness, and cropping. However, this depends on the nature of the dataset and the problem at hand. Ideally, experimenting with different methods iteratively to refine the model based on empirical results would be the best approach.

**References:**<br/>
1.	Krizhevsky, A., Sutskever, I. and Hinton, G.E. (2017) ‘ImageNet classification with deep convolutional Neural Networks’, Communications of the ACM, 60(6), pp. 84–90.
2.	Simonyan, K. and Zisserman, A., 2014. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
3.	Kim, J., Lee, J.K. and Lee, K.M., 2016. Accurate image super-resolution using very deep convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1646-1654).

