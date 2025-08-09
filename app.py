from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from sentiment_analysis_text import analyze_sentiment, analyze_subjectivity_polarity
from sentiment_analysis_image import SentimentAnalysisModel, load_model
from sentiment_analysis_audio import livePredictions
from style_transfer import transfer_style, pipelines

global emotion

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/sentiment_analysis', methods=['GET', 'POST'])
def sentiment_analysis():
    if request.method == 'POST':
        modality = request.form['modality']
        if modality == 'text':
            text = request.form['text']
            sentiment, score = analyze_sentiment(text)
            subjectivity, polarity = analyze_subjectivity_polarity(text)
            return render_template('text_sentiment_analysis.html', text=text, sentiment=sentiment, score=score, subjectivity = subjectivity, polarity = polarity)
        elif modality == 'image':
            image = request.files['image']
            filename = secure_filename(image.filename)

            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            
            model = SentimentAnalysisModel()
            loaded_model = load_model("img_sentiment.pkl")
            sentiment = loaded_model.analyze_sentiment(image_path)
            return render_template('image_sentiment_analysis.html', image_path=image_path, sentiment=sentiment)
        
        elif modality == 'audio':
            audio = request.files['audio']
            filename = secure_filename(audio.filename)
            
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio.save(audio_path)
            
            print(f"Audio file saved at: {audio_path}")  
            
            pred = livePredictions(path='SER_model.h5')
            pred.load_model()
            emotion = pred.makepredictions(audio_path)
            
            print(f"Predicted emotion: {emotion}")  
            
            return render_template('audio_sentiment_analysis.html', audio_path=audio_path, emotion=emotion)
        
    return render_template('sentiment_analysis.html')

@app.route('/text_sentiment_analysis', methods=['POST'])
def text_sentiment_analysis():
    text = request.form['text']
    sentiment, score = analyze_sentiment(text)
    subjectivity, polarity = analyze_subjectivity_polarity(text)
    return render_template('text_sentiment_analysis.html', text=text, sentiment=sentiment, score=score, subjectivity=subjectivity, polarity=polarity)

@app.route('/style_transfer', methods=['GET', 'POST'])
def style_transfer():
    if request.method == 'POST':
        text = request.form['text']
        style = request.form['style']
        output_text = transfer_style(text, style)
        return render_template('style_transfer_result.html', input_text=text, output_text=output_text)
    return render_template('style_transfer.html', styles=list(pipelines.keys()))

if __name__ == '__main__':
    app.run(debug=True)