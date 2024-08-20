PROJECT TITLE: DEVELOPMENT OF RADAR VIEW GENERATOR (GAN BASED)

This project represents a comprehensive and meticulous exploration into the integration of Generative Adversarial Networks (GANs) to advance RADAR technology applications. It began with a meticulous phase of data collection using RADAR sensors, ensuring the acquisition of a diverse and representative dataset containing essential classes such as pedestrians, bicycles, and cars. This dataset served as the foundational resource for training and evaluating three distinct GAN architectures: Conditional GANs, Deep Convolutional GANs, and Semi-Supervised GANs.


INTRODUCTION TO GANS AND THEIR MODEL: 

1) Conditional GAN: 
A Conditional Generative Adversarial Network (cGAN) is an extension of the conventional Generative Adversarial Network (GAN) framework that introduces additional conditional information into both the generator and discriminator architectures. This supplementary information, which can include class labels, textual descriptions, or any other relevant features, serves to guide and constrain the image generation process. Unlike traditional GANs that generate images solely from random noise, cGANs leverage this conditioning data to produce images that align with specified attributes or characteristics.
In cGANs, the generator receives bth random noise and conditional information as inputs, enabling it to generate images that adhere to the provided conditions. Meanwhile, the discriminator assesses the authenticity of generated images relative to both real data and the conditioning information. This dual input structure enhances the model's ability to learn and generate images that are not only visually realistic but also conform to desired attributes encoded in the conditioning data.

![image](https://github.com/user-attachments/assets/7bdcb739-bac3-4d26-9dd8-0c6829e3213e)


2) Deep Convolutional GAN (DCGAN)
A Deep Convolutional Generative Adversarial Network (DCGAN) is an advanced variant of the Generative Adversarial Network (GAN) that employs deep convolutional neural networks (CNNs) to enhance the fidelity of generated images. The defining characteristics of DCGANs include utilizing convolutional layers without pooling operations, incorporating batch normalization, and applying specific activation functions such as ReLU and Tanh. By integrating CNNs in both the generator and discriminator architectures, DCGANs achieve greater stability during the training process and produce higher-quality images than traditional GANs. This methodology marked a significant milestone in leveraging deep learning techniques to advance the performance and reliability of GANs.

![image](https://github.com/user-attachments/assets/10fafb81-f826-4ebd-8c2a-6f97f38897dd)

3) Semi-Supervised GAN (SSGAN)
The Semi-Supervised Generative Adversarial Network (SSGAN) amalgamates GAN principles with semi-supervised learning strategies to enhance classifier performance, particularly in scenarios where labelled data is limited. This hybrid approach harnesses the discriminator's dual role: distinguishing between real and synthetic images and classifying real images into predefined categories based on available labels.
In SSGAN, the discriminator undergoes training aimed at both discriminating real from fake images and accurately classifying real images into their respective classes using the limited labelled dataset. This supervised learning component is complemented by the use of unlabelled data, which aids in capturing the underlying data distribution through unsupervised learning techniques.

![image](https://github.com/user-attachments/assets/296b0732-fbbd-4ddb-a5ff-0696d9ba8fea)


DATASET:

The dataset is highly confidential and cannot be shared.


RESULTS: 

The following results shows the fake images created by generator of GAN model.

1) Conditional GAN:

![image](https://github.com/user-attachments/assets/da92ac8a-f0a3-4727-aa6c-26b7aa3fda6b)


2) Deep Convolutional GAN (DCGAN):
   
![image](https://github.com/user-attachments/assets/cc69066f-c8be-4979-aa57-b37a708e202f)


3) Semi-Supervised GAN (SSGAN):

![WhatsApp Image 2024-08-20 at 23 54 07_7ad35aef](https://github.com/user-attachments/assets/0ea571fd-f18d-4f40-a177-e7670ebfb33f)
