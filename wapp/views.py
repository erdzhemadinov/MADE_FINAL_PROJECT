import os
import sys

from flask import render_template

from wapp import app, predict, EXTRACTOR, EMBEDDER, interpretation #,parser
from wapp.forms import QAForm, UrlForm

@app.route('/')
@app.route('/index', methods = ['GET', 'POST'])
def index():
    return render_template("index.html")


@app.route('/find', methods = ['GET', 'POST'])
def login():

    qaForm = QAForm()
    urlForm = UrlForm()

    if qaForm.validate_on_submit():
        question = qaForm.question.data
        answer = qaForm.answer.data
        # Тут нужно вызвать модель
        prob =predict(1, 1, question, answer)
        return render_template('answerqa.html',
            question = question,
            answer = answer,
            prob = prob)

    if urlForm.validate_on_submit():
        # тут пока заглушка, надо распарсить страницу на вопросы и ответы по урлу
        # потом сгенерить html и показать результат
        return render_template(
            'answerurl.html', url = urlForm.url.data
        )
    
    return render_template(
        'find.html', title = 'Sign In', qaForm = qaForm, urlForm = urlForm
    )

@app.route("/interpretation", methods=['GET', 'POST']) 
def get_interpretation():
    qaForm = QAForm()
    question = qaForm.question.data
    answer = qaForm.answer.data
    print("q =",question)
    print("a =",question)
    interpretation(question, answer)
    return render_template("interpretation.html")