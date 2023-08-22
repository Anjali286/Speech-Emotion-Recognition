#Importing libraries
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import speech_recognition as sr
import tensorflow as tf

# Load emotion classification model and tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set up speech recognition
recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print('Clearing background noise...')
    recognizer.adjust_for_ambient_noise(source, duration=1)
    print('Waiting for your message...')
    recordedaudio = recognizer.listen(source)
    print('Done recording..')

try:
    print('Printing the message..')
    text = recognizer.recognize_google(recordedaudio, language='en-US')
    print('Your message:', text)
    
    # Perform emotion classification
    inputs = tokenizer(text, return_tensors="tf", padding="max_length", truncation=True, max_length=128)
    logits = model(**inputs).logits
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
    
    # Mapping of emotion labels
    emotion_labels = ["anger", "fear", "joy", "love", "sadness", "surprise"]
    
    # Print emotion probabilities
    for emotion, probability in zip(emotion_labels, probabilities):
        print(f"{emotion}: {probability:.4f}")
        
except Exception as ex:
    print(ex)

