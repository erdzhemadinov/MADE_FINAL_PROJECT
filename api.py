'''
Кусок кода с новым методом /getpreds просто засунуть в текущий вебсервис
'''

from flask import Flask
from flask import request
import requests as r
from flask.json import jsonify
import numpy as np


APP = Flask(__name__)


def make_preds(q, ans):
    '''
    q - это текст вопроса
    ans - это list ответов на него
    return - list с вероятностями принадлежать ответа к троллингу 
             (порядок должен быть такой же как в ans)
    '''

    # тут нужно вызывать модель вместо заглушки
    return list(np.random.random(len(ans)))



@APP.route('/getpreds', methods=['POST'])
def getpreds():

    jdict = request.get_json(force=True)['items']
    answer = {'response': []}
    for pair in jdict:
        q = pair['question']
        ans = pair['answers']
        answer['response'].append({'preds': make_preds(q, ans) })
    return jsonify(answer)

# Запускаем веб-сервис

if __name__=='__main__':
    APP.run()
    '''
    Тестить сервис так. Стандартный вид jsona такой:
    j = {
        'items': [
            {'question': 'q1', 'answers': ['a11', 'a12', 'a13', 'a14']}
            , {'question': 'q2', 'answers': ['a21', 'a22']}
            , {'question': 'q3', 'answers': ['a31']}
        ]
    }
    a = r.post('http://127.0.0.1:5000/getpreds', json=j)
    print(a.text)

    Минимальный json (1 вопрос с 1 ответом)
    j = {
        'items': [
            {'question': 'ывапывпыв', 'answers': ['ыфвпфвыпвыфп']}
        ]
    }
    a = r.post('http://127.0.0.1:5000/getpreds', json=j)
    print(a.text)
    '''