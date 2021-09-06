from flask import Flask
from numpy import load

# from parsers import get_question_answers
from random import random 
app = Flask(__name__)
app.config['WTF_CSRF_ENABLED'] = False

EXTRACTOR = 1
EMBEDDER = 1

# For interpretation

def interpretation(question, answer):
    return 1

def predict(emb1, emb2, question, answer):
    return random()

from wapp import views
