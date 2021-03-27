Part A

Framework and Libraries

In this coursework, I used PyTorch as the framework to help me to build different neural networks and carry out the whole deep learning process. The reason that I chose PyTorch is that I noticed that building machine learning models with PyTorch feels more intuitive and is simple to use which makes the learning curve relatively short. PyTorch’s syntax is very similar to Python itself, models, layers and many other things like optimizers and data loaders are simply python classes. Although I haven’t used other deep learning frameworks before, after looked into some example codes online, I felt that PyTorch will be an easier option for deep learning beginner like myself. 
For libraries, there are two essential ones that have a huge contribution to the coursework. Torchvision is part of the PyTorch framework, it consists of some popular datasets including the MNIST dataset and some model architectures. With torchvision, I was able to download and use the dataset effortlessly with just a few lines of code. The other one is matplotlib, a plotting library. It makes visualising results so simple, again, with just a few lines of code, I was able to plot informative graphs showing how well a model is performing compared to others.

Learning

Since this is my first time using PyTorch, I spent some time looking for tutorials that teach how to use Pytorch to undertake a deep learning project. Because of the popularity of the framework, I was able to find a lot of articles and even videos for it. There is one particularly useful online course that I used to gain a sufficient understanding of the framework and the deep learning project pipeline [1]. From this course, I have learnt how to use PyTorch and the overall pipeline of the project, for example, the steps needed to build a deep learning model, then train it and finally get some results of the model. In addition, I have used the PyTorch official documentation and tutorials to help me to build my deep learning models [2]. Due to the simplicity of PyTorch, it was not difficult to derive my own models’ architecture. The main difference between my models’ architecture and the ones from the tutorials was the layers’ structure.

Modification

To build my own model architectures, I mainly focus on modifying the layers of the models. In my code, there are two fully connected neural networks and one convolutional neural network. I built these models by referencing the example models in the tutorial and then I modified the number of layers and hidden nodes. While there are not much to modify in a fully connected layer network, there are much more that can be modified in a convolutional network. Other than number of layers, there are kernel size, stride, padding and pooling layer that can have countless combination. I did configure my own version of a convolutional network by changing these variables and its performance on the MNIST dataset is brilliant.

Experiment Methods

To compare the three models, I used accuracy to be the main index. The accuracy of a model is calculated by the number of correctly identified items divided by the total number of items in a set. I chose to use accuracy as the indicator of the models’ performance because it is easier to understand as everyone knows what accuracy is, a person doesn’t need to be a data scientist to understand how well the model is performing. Also, I have kept track of the loss of the models, which indicates how bad a model’s prediction is, as a second index indicating the performance of the models. And this is calculated by using the cross-entropy loss function, which is the most common one for classification problems [3]. Again, with the help of the PyTorch library, cross-entropy can be calculated with a single line of code, which is very handy and yet very useful.
In order to run a fair experiment, I decided to change only one hyperparameter at a time to see what effect does changing a particular hyperparameter will bring to the models. Throughout the experiment, only epochs, optimizer and learning rate have been changed, therefore, my experiment is only showing how these three parameters are affecting the models.  With three model architectures, I have run 12 tests with each of them and I got 36 results in total at the end. As mentioned above, I changed one of the three hyperparameters at a time to see what effects that bring to the models. For epochs, I tried 10 and 20. For the optimizer, I tried Adam and SGD and for the learning rate, I tried 0.01, 0.05 and 0.1. In addition, I have also recorded the training time of a model for every test run just to see how much time the models need to complete training.

Part B

Overall Results

From the results I got, it is very clear that the CNN model was always the one with the highest accuracy and lowest average loss when running on the validation set and test set comparing to the other two fully connected models. Among all models and all the combinations of parameters in the experiment, the one with the highest accuracy and the lowest average loss on the test dataset is the CNN model with two convolutional layers and one linear layer trained for 20 epochs with SGD optimizer and a learning rate of 0.1. After that, it is often that the fully connected model with 1 hidden layer achieved the second-highest accuracy and second-lowest loss. And finally, the one with 3 hidden layers is always outperformed by the other two models.

Epochs

After training the models with different number of epochs, I found that number of epochs does not have a significant impact on a model’s accuracy. For example, the accuracy at the 10th epoch and the 20th epoch are similar (figure 2). Perhaps, 20 epochs are not enough to make any difference. Therefore, I ran some more test on some of those which didn’t have an accuracy above 70%. I tried to train them with 50 epochs and see if that will increase the accuracy. While some results showed that even with 50 epochs, the accuracies were still similar, some results were obviously indicating the models were overfitted. However, when I compared the validation loss between the same model with the same configuration but different epochs, it was noticeable that the numbers dropped when the model was trained with more epochs (figure 3).

Optimizers

I used two different optimizers to train my models, SGD and Adam. With the results I got (figure 1), I will say SGD is a better option comparing to Adam when it is used to train for the MNIST dataset. The results produced using SGD not only have higher accuracy and lower losses, but the results are also more consistent. If we look at the results produced by models that trained with Adam, the accuracy seems a bit off sometimes. Also, I noticed that the Adam optimizer seems to work better with a lower learning rate. With a 0.01 learn rate, models with Adam optimizer tend to yield higher accuracy and lower loss compared to the other configurations of themselves. 

Learning Rate

For the majority of the test runs, the three models’ accuracy always settled within 5 epochs. I deduce that I have set the learning rates a bit too high so that the accuracy jumped right to the optimal value. If we look into a particular set of results where the models were trained with SGD optimizer with 0.01 learning rate, the lowest rate I used, we can see the accuracy fluctuated a bit before it settled (figure 2). And this cannot be found in results that have a higher learning rate. I believe the fluctuation indicating the models were finding the optimal. This behaviour indicates that the learning rate is in fact, affecting the speed for a model to learn. With a lower learning rate, the model will take longer to find its optimal by adjusting the weights and biases bit by bit. 

Conclusion

For this experiment, one thing that I didn’t expect was the fully connected network with 1 hidden layer will outperform the one with 3 hidden layers. And because of that, I have learnt that more hidden layers do not mean a better model. The number of layers and the number of nodes should be adjusted according to the input and output size [4]. Although a model trained with more epochs doesn’t mean that it will have higher accuracy, with more epochs, the model should have a lower loss. When comparing SGD and Adam optimizer, SGD has shown better results in this experiment with the MNIST dataset. I believe the performance of the two optimizers will differ when they are used for different datasets. And the results are showing that Adam works better with lower learning rates. For learning rates, a lower rate means the model will take longer to find its optimal and have a lower chance to miss the optimal. Therefore, it is convincing that when training a neural network, starting with a low learning rate is a proper choice. After all, the best result I generated is the CNN model with 2 convolutional layers and one fully connected layer, trained for 20 epochs with the SGD optimizer at a learning rate of 0.1.  I believe the way to find an optimal model for any dataset is by trial and error, but with the understanding of the relationship between the parameters and the performance, it will be easier to adjust the model and achieve an optimal configuration.

References

[1] AAKASH, N.S. Deep Learning with PyTorch: Zero to GANs[online]. Jovian, 2020. [viewed 22/02/21]. Available from: https://jovian.ai/aakashns/01-pytorch-basics

[2] SURAJ, S., SETH, J. and CASSIE, B.  Learn the Basics[online]. PyTorch, 2017. [viewed 22/03/21]. Available from: https://pytorch.org/tutorials/beginner/basics/intro.html

[3] RAVINDRA, P. Common Loss Functions in Machine Learning[online]. Towards Data Science, 2018. [viewed 25/02/21]. Available from: https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23

[4] WARREN, S. Comp.ai.neural.nets FAQ, Part 3 of 7: Generalization [online]. Faqs.org, 2014. [viewed 15/03/21]. Available from: http://www.faqs.org/faqs/ai-faq/neural-nets/part3/preamble.html
