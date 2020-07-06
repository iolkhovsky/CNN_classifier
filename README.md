## Digits CNN classifier

This 'helloworld' project implements image classifier intended to recognize hand-written digits. PyTorch project includes model (simple CNN) definition, train and test scripts. Classifier has been trained and tested on MNIST dataset. Accuracy after 1 epoch achieved 97%.

![Alt text](attachments/valid_sample_final.png?raw=true "Result")

Scripts have been tested with torch v1.5.1, opencv v4.4.0-pre, sklearn v0.23.1, numpy v.1.19.0 under Ubuntu 18.04

#### Dataset

Standard torchvision.datasets.MNIST dataset is used

#### Model

Model's architecture defined in cnn_classifier.py

#### Training

Example of training script running is shown down below 

![Alt text](attachments/training.png?raw=true "Train")

Training script uses Tensorboard to visualize training process

![Alt text](attachments/tensorboard_loss.png?raw=true "TB loss")
![Alt text](attachments/tensorboard_accuracy.png?raw=true "TB acc")
![Alt text](attachments/tensorboard_images.png?raw=true "TB acc")

#### Testing

Example of test script running is shown down below

![Alt text](attachments/testing.png?raw=true "Test")


#### Simple OCR using model

There is ocr_demo.py script, which shows example of toy OCR project based on the trained model

![Alt text](attachments/ocr_result.jpg?raw=true "ocr")

