# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from flask import Flask
from flask_script import Manager
from flask_sqlalchemy import SQLAlchemy
import bcrypt

import os
import sys
from numpy import load as np_load
from numpy import round_
from random import random
from datetime import datetime

#________________________________________________________________________________________
flag_model = True
#________________________________________________________________________________________

path_= os.path.dirname(os.path.dirname(__file__))
paths_= []
paths_.append(os.path.join(path_, 'app'))
paths_.append(os.path.join(path_, 'app', 'models'))
paths_.append(os.path.join(path_, 'binary_class_results_interpretation'))
paths_.append(os.path.join(path_, 'models'))
paths_.append(os.path.join(path_, 'parsers'))

for p in paths_:
    if p in sys.path:
        pass
    else:
        sys.path.append(p)

if sys.platform == "win32":
    base_slash = "///"
else:
    base_slash = "////"

if flag_model:
    from models.trolldetector import IsTrollClassifierModel
    from binary_class_results_interpretation.explainResultsToHTML import ExplainResultsToHTML

from parsers.otvetmailparser import get_question_answers
from app.models.create_html_table import Table

# from test_plotly import polt_ploty

app_web = Flask(__name__)
app_web.config['TEMPLATES_AUTO_RELOAD'] = True
app_web.config['SECRET_KEY'] = 'a really really really really long secret key'
app_web.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:'+base_slash+\
    os.path.join(path_, 'app', 'database','database.db')

manager = Manager(app_web)
db = SQLAlchemy(app_web)

class Users(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.BigInteger(), primary_key=True)
    email = db.Column(db.String(), nullable=False)
    password = db.Column(db.String(), nullable=False)
    created_on = db.Column(db.DateTime(), default=datetime.utcnow)
    updated_on = db.Column(db.DateTime(), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    api = db.relationship('ApiInformation', backref='user')
    history = db.relationship('UsersHistory', backref='user')

    def __repr__(self):
	    return "<{}:{}>".format(self.id,  self.email)

    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password, bcrypt.gensalt())

    def check_password(self,  password):
        return bcrypt.checkpw(password, self.password_hash)


class ApiInformation(db.Model):
    __tablename__ = 'api_information'
    api_key = db.Column(db.String(), primary_key=True, nullable=False)
    tarif = db.Column(db.String(), nullable=False)
    counts_proba_left = db.Column(db.BigInteger(), nullable=False)
    counts_link_left = db.Column(db.BigInteger(), nullable=False)
    counts_interpr_left = db.Column(db.BigInteger(), nullable=False)
    created_on = db.Column(db.DateTime(), default=datetime.utcnow)
    updated_on = db.Column(db.DateTime(), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user_id = db.Column(db.Integer(), db.ForeignKey('users.id'))

    def __repr__(self):
	    return "<{}:{}>".format(self.api_key, self.tarif)

    def create_apy_key(self, login):
        bcrypt.kdf(
            password=b'key_api_login:'+login,
            salt=b'salt',
            desired_key_bytes=32,
            rounds=10)

class UsersHistory(db.Model):
    __tablename__ = 'users_history'
    id = db.Column(db.BigInteger(), primary_key=True)
    question = db.Column(db.Text(), nullable=False)
    answer = db.Column(db.Text(), nullable=False)
    proba = db.Column(db.Float(), nullable=False)
    created_on = db.Column(db.DateTime(), default=datetime.utcnow)
    updated_on = db.Column(db.DateTime(), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user_id = db.Column(db.Integer(), db.ForeignKey('users.id'))

    def __repr__(self):
	    return "<{}:{}>".format(self.id,  self.question[:20])

class AllHystory(db.Model):
    __tablename__ = 'history'
    id = db.Column(db.BigInteger(), primary_key=True)
    question = db.Column(db.Text(), nullable=False)
    answer = db.Column(db.Text(), nullable=False)
    proba = db.Column(db.Float(), nullable=False)
    created_on = db.Column(db.DateTime(), default=datetime.utcnow)
    updated_on = db.Column(db.DateTime(), default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
	    return "<{}:{}>".format(self.id,  self.question[:20])

db.create_all()

TABLE = Table()
TABLE_HISTORY = Table(columns=['Вопрос', 'Ответ', 'Вероятность','Время запроса'], max_size_rows=150, reverse_data=True)
TABLE_INTERP = Table(columns=["Признак", "Название признака", "Значение признака", "Среднее значение по датасету"], with_dooble_click_script=False)
TABLE_INTERP_TOP = Table(columns=["Признак", "Доля вклада","Признак", "Доля вклада"], with_dooble_click_script=False)
MAX_TOP_IMPACT = 5

if flag_model:
    MODEL = IsTrollClassifierModel()
    MODEL.load()
    EMBEDDER = MODEL.embedder
    EXTRACTOR = MODEL.extractor
    IS_TROLL = MODEL.is_troll

    FEATURES_TROLL = np_load("models/features/troll_features.npy")
    FEATURE_SCALE = IS_TROLL.scaler.transform(FEATURES_TROLL)
    EXPL = ExplainResultsToHTML(
            model=IS_TROLL.logreg, X_train=FEATURE_SCALE, model_type='linear',
            is_proba=True, scaler=IS_TROLL.scaler
        )

    FEATURES_NAME = [
                    "q_toxic", "q_abs_neg", "q_neg", "q_neutral", "q_pos", "q_abs_pos",
                    "q_threat", "q_word_count", "q_char_count", "q_world_lenght", "q_stop_word_count", "q_digit_count", "q_title_word_count", "q_url_count",
                    "q_emo_type", "q_emb",

                    "a_toxic", "a_abs_neg", "a_neg", "a_neutral", "a_pos", "a_abs_pos",
                    "a_threat", "a_word_count", "a_char_count", "a_word_lenght", "a_stop_word_count", "a_digit_count", "a_title_word_count", "a_url_count",
                    "a_emo_type", "a_emb",

                    "qa_cos_sim", "qa_is_joke", "qa_is_best"
                    ]

    FEATURES_NAME_FULL = [
                    "Вероятность токсичности вопроса",
                    "Вероятность абсолютной негативности вопроса",
                    "Вероятность негативности вопроса",
                    "Вероятность нейтральности вопроса",
                    "Вероятность позитивности вопроса",
                    "Вероятность абсолютной позитивности вопроса",
                    "Вероятность содеражния угрозы в вопросе",
                    "Количество слов в вопросе",
                    "Количество символов в вопросе",
                    "Средняя длина слова в вопросе",
                    "Количество стоп-слов в вопросе",
                    "Количество цифр (чисел) в вопросе",
                    "Количество слов с заглавной буквы в вопросе",
                    "Количество ссылок в вопросе",
                    "Эмоциональный тип вопроса",
                    "Эмбеддинг вопроса",
                    
                    "Вероятность токсичности ответа",
                    "Вероятность абсолютной негативности ответа",
                    "Вероятность негативности ответа",
                    "Вероятность нейтральности ответа",
                    "Вероятность позитивности ответа",
                    "Вероятность абсолютной позитивности ответа",
                    "Вероятность содеражния угрозы в ответе",
                    "Количество слов в ответе",
                    "Количество символов в ответе",
                    "Средняя длина слова в ответе",
                    "Количество стоп-слов в ответе",
                    "Количество цифр (чисел) в ответе",
                    "Количество слов с заглавной буквы в ответе",
                    "Количество ссылок в ответе",
                    "Эмоциональный тип ответа",
                    "Эмбеддинг ответа",
                    
                    "Косинусное сходство вопроса и ответа",
                    "Вероятность шуточного ответа на вопрос",
                    "Вероятность полезности ответа на вопрос",
                    ]

    def interpretation_top(features_scale, n_max, draw_table=True, draw_pics=True):
        neg = EXPL.get_impact_of_n_max_shap_values(features_scale, FEATURES_NAME_FULL, n_max, is_pos=False)
        pos = EXPL.get_impact_of_n_max_shap_values(features_scale, FEATURES_NAME_FULL, n_max, is_pos=True)
        
        if draw_pics:
            pass
            # polt_ploty(pos, "Позитивный", class_="is_troll", path=os.path.join(path_, 'app','templates',''))
            
            # EXPL.pie_plot_impacts_by_classes(pos, neg, 
            #     show_pics=False, 
            #     save_pics=True, 
            #     path_save=os.path.join(path_, 'app','static','pic',''),
            #     dpi_pic=20)

        if draw_table:
            neg_items = list(neg.items())
            pos_items = list(pos.items())
            neg_items.pop(list(neg.keys()).index("Остальное"))
            pos_items.pop(list(pos.keys()).index("Остальное"))
            neg_len = len(neg_items)
            pos_len = len(pos_items)

            TABLE_INTERP_TOP.clear_table()

            for i in range(max(neg_len, pos_len)):
                if i < neg_len and i < pos_len:
                    TABLE_INTERP_TOP.add_row([pos_items[i][0], 
                        round(pos_items[i][1],3), 
                        neg_items[i][0], 
                        round(neg_items[i][1],3)])
                elif i < neg_len and i >= pos_len:
                    TABLE_INTERP_TOP.add_row(["", "", neg_items[i][0], round(neg_items[i][1],3)])
                else:
                    TABLE_INTERP_TOP.add_row([pos_items[i][0], round(pos_items[i][1],3), "", ""])
            # TABLE_INTERP_TOP.save_html(path="app/templates/table_interpretation_top.html",
            #                                 table_id="table_top",
            #                                 class_table="table_top")
        return neg, pos
        
    def interpretation_short(question, answer, n_max_top=5):
        qa_emb = EMBEDDER.encode(" ".join((question, answer)))
        q_emb = EMBEDDER.encode(question)
        a_emb = EMBEDDER.encode(answer)
        features = EXTRACTOR.create_features(
            question, q_emb, answer, a_emb, qa_emb
        )
        feature_scale = IS_TROLL.scaler.transform(features)
        return interpretation_top(feature_scale[0], n_max_top,  draw_table=False, draw_pics=False)

    def interpretation(question, answer, solve_plot=True, solve_table=True):
        qa_emb = EMBEDDER.encode(" ".join((question, answer)))
        q_emb = EMBEDDER.encode(question)
        a_emb = EMBEDDER.encode(answer)
        features = EXTRACTOR.create_features(
            question, q_emb, answer, a_emb, qa_emb
        )
        feature_scale = IS_TROLL.scaler.transform(features)
        interpretation_top(feature_scale[0], MAX_TOP_IMPACT)
        if solve_plot:
            EXPL.single_plot(
                FEATURES_NAME,
                feature_scale,
                path_save="app/templates/single_plot.html")
        if solve_table:
            average_features = round_(FEATURES_TROLL.mean(axis=0), 4)
            round_features = round_(features[0], 4)
            TABLE_INTERP.clear_table()
            for i, name_feature in enumerate(FEATURES_NAME):
                TABLE_INTERP.add_row([name_feature, FEATURES_NAME_FULL[i], round_features[i], average_features[i]])
        # TABLE_INTERP.save_html(path="app/templates/table_interpretation.html")

else:
    class Test:
        def predict(self, question, answer):
            return round(random(),3)
    MODEL = Test()
    def interpretation(question, answer):
        pass

from app import views