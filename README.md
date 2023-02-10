# Smart Shape Recognition

This project presents a method for shape recognition using data generation and training. The method involves generating synthetic data of 3 different geometric shapes and then training a Convolutional Neural Network on this data to recognize and classify the shapes. The network is enriched with hand-drawn images, and the results show that the network is able to accurately recognize and classify shapes. This approach is tested with different models throughout the project via Streamlit interface. The aim is to utilize this method, particularly for Parkinson’s disease.

DATASETS

The first data set used in the project is the data set where perfect shapes were created. It was plotted according to the height and width values determined for triangles and squares. Circles were plotted according to the determined radius length. Data augmentation was performed on the plotted figures for triangles, squares, and circles. By generating the shapes that were originally created, 4000 shapes were obtained from each shape. These shapes are divided into 3 classes train, test, and validation. A rate of 80 percent was determined for the training process, and 10 percent for testing and validation. The classification process over batch is shown in Figure 1. 0 represent circles, 1 represent squares and 2 represent triangles.
![image](https://user-images.githubusercontent.com/33731743/218131779-8c647801-d20c-4d28-bac9-9a1c9b07fd5c.png)

The second data set is created by merging two data sets. In addition to the first perfect shape data set, an existing data set Geometric Shapes Mathematics from Kaggle was added to improve the understanding of hand-drawn shapes. A total of 6000 shapes are presented in the data set. These shapes are divided into 3 classes train, test, and validation. A rate of 80 percent was determined for the training process, and 10 percent for testing and validation.
![image](https://user-images.githubusercontent.com/33731743/218131994-14e9c5ff-1029-4cc1-806b-e203526e4f17.png)

TRAINING AND VALIDATION

Below, the accuracy and loss graphs for each model are presented. The X-axis represents the number of epochs, the Y-axis represents accuracy or loss. The accuracy of models increased with each epoch on the training data set. In some cases, when training accuracy continues to increase the validation accuracy decreases. This is a sign of overfitting, models tend to memorize the training data instead of generalizing to new data. Overfitting is most common in the graph of the 3 convolution layer perfect shapes model. It is understandable because overfitting occurs when the machine learning model is unable to generalize its predictions to new data and fits too closely to the training dataset.

![image](https://user-images.githubusercontent.com/33731743/218132413-80556c51-a43b-4e03-815d-1dcbd53ce2c6.png)

The 5 convolutional layers of CNN have the ability to learn the underlying patterns in the data better than the 3 convolutional layers CNN, which results in higher accuracy, due to the additional layers in the model providing more capacity for learning. Because the 5 convolutional layers CNN has a higher number of parameters to optimize, it will take a longer time to train in comparison to the 3 convolutional layers CNN.

![image](https://user-images.githubusercontent.com/33731743/218132568-16a6f67e-ebe0-462f-a9ae-5ee3e7453b91.png)

5 convolutional layers CNN and 3 convolution layer CNNmixed data set gave very close accuracy results, but 3 layer mixed data set’s accuracy is slightly less than 5 convolution layer CNN. With compared with 3 convolution layer CNN overfitting occurred much less in 3 convolution layer CNNmixed shapes even though it has 3 layers too. When we fed the data set in more diverse ways, it became easier to learn new data.

![image](https://user-images.githubusercontent.com/33731743/218132733-2cca471e-fb27-4090-9395-e7750f345e81.png)

INTERFACE

Streamlit was chosen to create an easy interface for users. It is an open-source Python library that allows developers to create web apps for machine learning and data science. A sidebar has been added to our interface so that users can easily try different models. Three different models are embedded in each sidebar. A short function was written into these sidebars to get prediction output. Each sidebar has a drawable canvas inside. The drawable canvas is added to the Streamlit application using the ”st.canvas” function. The canvas is essentially a blank space on which the user can draw interactively. In this way, users can draw shapes and try on models in real-time.

After the training was carried out on the 3 models created, the testting was started over the interface. As shown in figure, the similarity percentage of the
square drawn on the canvas in the 3 convolutional layers model trained with the perfect shape data set was 39.20.

![image](https://user-images.githubusercontent.com/33731743/218133154-84e6a355-d5be-4c29-ac9a-01d986ab0702.png)

The similarity rate of the second model with 5 layers trained with perfect shapes is 99.90.

![image](https://user-images.githubusercontent.com/33731743/218133257-dda29e96-a1af-45e9-ad3b-f3747b609c4d.png)

The similarity rate of the third and final model with 3 layers trained with mixed shapes is 100.

![image](https://user-images.githubusercontent.com/33731743/218133352-4f7e0e85-e94f-4f4f-8539-02bf0235f762.png)

CONCLUSION

In conclusion, the Smart Shape Recognition project was a success in using data generation and training three different models to accurately identify three shapes. This article provided a comparison of models and testing with real-life examples. The implementation of Streamlit as the user interface made the process of classifying shapes user-friendly and efficient. The results show how different models performed in identifying the hand-drawn shapes. And the data generation techniques used were effective in creating a diverse data set for training. Overall, the project demonstrated the effectiveness of utilizing machine learning techniques for shape recognition and the potential for further improvement in accuracy and efficiency. The further aim of this project would be to use this for Parkinson’s disease since Parkinson’s disease have difficulty with fine motor skills, such as drawing.

