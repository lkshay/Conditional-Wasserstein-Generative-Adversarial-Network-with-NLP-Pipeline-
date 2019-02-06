### Python files for generating images with input text descriptions by the user...


# Prereqs:
1) FastText Word Embeddings (Facebook Research) for NLP.
These word embeddings can be trained for any applications provided input context and output labels. The vectors (words in feature space) can be clustered according to any context (as depicted by the dataser). Here, the input descriptions are some text samples like "Show me something that is parked somewhere on the road" and the output labels can possibly be a Car or a Truck. 
2) TensorFlow 
3) Other common python tools including OpenCV

# The Pipeline is as follows: 
One of the project_demo.sh file can be used to enter the text descriptions in the terminal. The FastText word embeddings are trained to produce one of the 10 classes in Cifar10 dataset. After this label is produced, it is run through the DCGAN to produce an artificial image of the class label.

The network was trained on a machine with GPU and is a long process as we are training multiple neural networks (a Convolutional Nueral Network (Descriminator) and Up sampling (Generative) network). 

Details about the network architecture can be seen in the PDF report. Implementation can be seen in nets.py

References can also be seen in the report.

Tested in Ubuntu 18.04
