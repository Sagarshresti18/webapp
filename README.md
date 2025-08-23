Multimodal Sentiment Analysis Web App
🎯 Overview
This web application performs sentiment analysis across three modalities:
- 📝 Text: Analyzes sentiment, subjectivity, and polarity of user input.
- 🖼️ Image: Predicts sentiment from facial expressions using a trained image model.
- 🎙️ Audio: Detects emotion from voice recordings using MFCC features and a deep learning model.
 Features
- Text Sentiment Analysis: Uses NLP techniques to classify sentiment and measure subjectivity/polarity.
- Image Sentiment Analysis: Predicts emotion from facial images using a pre-trained model.
- Audio Sentiment Analysis: Extracts MFCCs and uses a Conv2D model to classify emotions like happy, sad, angry, etc.
- Style Transfer: Transforms input text into different writing styles using fine-tuned pipelines.
Notes
- The image sentiment model (img_sentiment.pkl) and audio model (SER_model.h5) must be present in the root directory.
- Uploaded files are stored in the uploads/ folder.
- Style transfer uses predefined pipelines for tone transformation.
Audio Input Guidelines
- Format: .wav
- Duration: 2–5 seconds
- Content: Clear emotional speech (e.g., happy, sad, angry)
- Sampling Rate: 22050 Hz
- Mono channel preferred
🛠️ Tech Stack
💻 Frontend
- HTML5 – Structure and layout
- CSS3 – Styling and responsive design
- Jinja2 – Templating engine for dynamic content rendering
🧠 Backend
- Python 3.12
- Flask – Lightweight web framework for routing and server logic
🧪 Machine Learning & NLP
- Keras – Deep learning framework for building and loading models
- Scikit-learn – Image sentiment model and preprocessing
- TextBlob – Text sentiment, polarity, and subjectivity analysis
🎙️ Audio Processing
- Librosa – Audio feature extraction (MFCCs)
- NumPy – Numerical operations and array manipulation
🖼️ Image Processing
- OpenCV (optional) – Image preprocessing and face detection
- Pillow – Image loading and manipulation
🔤 Style Transfer
- Custom Pipelines – Rule-based or model-driven text transformation
📦 Utilities
- Werkzeug – Secure file uploads
- OS – File system operations.

