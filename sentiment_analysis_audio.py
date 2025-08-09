import numpy as np
import keras
import librosa

# Define the livePredictions class for audio sentiment analysis
class livePredictions:
    """ Main class of the application. """
    def __init__(self, path):
        """ Init method is used to initialize the main parameters. """
        self.path = path

    def load_model(self):
        """ Method to load the chosen model. :param path: path to your h5 model. :return: summary of the model with the .summary() function. """
        self.loaded_model = keras.models.load_model(self.path)
        return self.loaded_model.summary()

    def makepredictions(self, input_file):
        """ Method to process the input file and create your features. """
        print("Selected file:", input_file)

        # Load audio file using librosa
        data, sampling_rate = librosa.load(input_file)

        # Extract MFCC features
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=0)
        x = np.expand_dims(x, axis=2)

        predictions = self.loaded_model.predict(x)
        predicted_class = np.argmax(predictions, axis=1)
        analyzed_emotion = self.convertclasstoemotion(predicted_class)
        return analyzed_emotion

    @staticmethod
    def convertclasstoemotion(pred):
        """ Method to convert the predictions (int) into human readable strings. """
        label_conversion = {0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad', 4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'}
        for key, value in label_conversion.items():
            if key == pred:
                label = value
        return label

# Example usage
if __name__ == "__main__":
    input_file = input("Enter the path to the audio file: ")
    pred = livePredictions(path='SER_model.h5')
    pred.load_model()
    analyzed_emotion = pred.makepredictions(input_file)
    print(f"Sentiment for the audio: {analyzed_emotion}")