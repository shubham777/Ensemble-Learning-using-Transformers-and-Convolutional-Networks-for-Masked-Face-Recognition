# Ensemble Learning using Transformers and Convolutional Networks for Masked Face Recognition
This repository contains the code of the paper **[Ensemble Learning using Transformers and Convolutional Networks for Masked Face Recognition](https://arxiv.org/abs/2210.04816)** by by Al-Sinan et al.
The paper suggests a novel solution to the problem of recognizing faces while wearing masks. Wearing a face mask was one of the changes we had to make to slow the spread of the coronavirus. The constant covering of our faces with masks has created a need to understand and investigate how this behavior affects the recognition capability of face recognition systems. When dealing with unconstrained general face recognition cases, current face recognition systems have extremely high accuracy, but they do not generalize well with occluded masked faces. The proposed method combines two popular deep learning techniques, Convolutional Neural Networks (CNNs) and Transformers, to produce an ensemble model capable of accurately recognizing masked faces. Two CNN model, fine-tuned on FaceNet pre-trained models extracts features from the face image, while two Transformer model learns the relationship between these features and the person's identity. An ensemble of these four models, using a majority voting technique was used to identify the individual whose face was covered by the mask. Experiment results showed that the proposed ensemble model outperforms existing state-of-the-art masked face recognition methods. The proposed system was tested on a synthetically masked LFW dataset generated in this study. The ensembled models provide the best accuracy, with a 92% accuracy. This recognition rate outperformed other models' accuracy, demonstrating the correctness and robustness of the proposed model for recognizing masked faces.

# Dataset
Please use this link provided by the authors to download the dataset:

https://drive.google.com/drive/folders/1uiTAWdU2DMyy27j3H6BPnLCiQvFRH5nD

# Modelling

## CNN models fine-tuned on FaceNet dataset 
Three CNN models fine-tuned on different pre-trained models (VGG16, EfficientNet, FaceNet) were studied. It was seen that the CNN models fine-tuned on FaceNet pre-trained model provided the best accuracy and hence, this model was selected for the for the final classification task. Since, FaceNet was trained only on images of maskless people and the aim of the project is to detect images of masked faces, the last layers of this model have been modified and a dropout and batch normalization layers were added to this model. The output features of this model were fed into the classification layer. A Softmax function in the classification layer was used, which assigns a probability for each predicted subject.

![Alt text](cnn_models.png?raw=true)

## Transformer models 
A Transformer model, using only the encoder component was also used for the masked face recognition task. The proposed Transformer, which was inspired by the work of [Dosovitskiy et al]([url](https://arxiv.org/abs/2010.11929)), accepts an input image and splits it into a number of patches. Each patch represents one time step and is fed into the encoder. However, the spatial information of the input image is not preserved by this image segmentation. To address this issue, each patch uses position embedding to encode its position in the original input image. The resulting patch with its embedded position is fed into the encoder of the Transformer. The encoder is made up of a layer normalization unit and a multi-head attention (MHA) unit. A residual connection is also used with the MHA unit, and the connection's output is fed into another normalization layer, which is followed by a multi-perception layer (MPL). MPL is made up of two blocks of different configurations of fully connected (FC) and dropout layers. The MPL output is combined with the MHA output to form the transformer output.
![Alt text](transformer.png?raw=true)

## An Ensemble of two CNN and two Transformer models
In this paper, two approaches to Ensemble Learning were used. The first method entails creating separate validation sets for each model. This method aids in overcoming the overfitting issue of the models involved. The second method is to combine various models with various configurations. This facilitates the use of these models' capabilities for masked face identification. Furthermore, combining the predictions of multiple models can reduce the variance of the resulting ensembled model.
![Alt text](ensemble.png?raw=true)




# Results

| Model | Top-1 Accuracy | Top-5 Accuracy
| -- | -- | -- |
CNN-VGG16 | 73.38% | 82.05% |
CNN-EfficientNet | 79.61% | 84.41% |
CNN-FaceNet | 80.30% | 85.24% |
Transformer | 69.04% | 78.70% |
Ensemble Learning | **92.01%** | **96.57%** |
