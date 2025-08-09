import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# Define the emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define the CNN model for image sentiment analysis
class SentimentAnalysisModel(nn.Module):
    def __init__(self):
        super(SentimentAnalysisModel, self).__init__()
        # Load pre-trained ResNet-18 model
        self.resnet = models.resnet18(pretrained=True)
        # Replace the first convolutional layer to accept grayscale input
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace the final fully connected layer with a new one for our task
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, len(emotions))

    def forward(self, x):
        return self.resnet(x)

    def preprocess_image(self, image_path):
        # Load and preprocess the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert RGB to grayscale
        image = cv2.resize(image, (48, 48))
        image = transforms.ToPILImage()(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.5], std=[0.5])(image)
        # Repeat the grayscale image to create 3 channels
        image = torch.cat((image, image, image), dim=0)
        return image

    def analyze_sentiment(self, image_path):
        # Preprocess the image
        image = self.preprocess_image(image_path)
        image = image.unsqueeze(0)

        # Make prediction using the loaded model
        with torch.no_grad():
            output = self(image)
        _, predicted = torch.max(output.data, 1)
        predicted_label = emotions[predicted.item()]
        return predicted_label

# Load the saved model from the pickle file
def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Example usage
if __name__ == "__main__":
    image_path = input("Enter the path to the image: ")
    if not os.path.isfile(image_path):
        print("Image file does not exist.")
        exit()

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    model = SentimentAnalysisModel()
    loaded_model = load_model("img_sentiment.pkl")
    sentiment = loaded_model.analyze_sentiment(image_path)
    print(f"Sentiment for the image: {sentiment}")