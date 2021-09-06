# -*- coding: utf-8 -*-
from flask import request, render_template, make_response, jsonify
from app import app_web, MODEL, interpretation, get_question_answers, interpretation_short, TABLE, TABLE_HISTORY, TABLE_INTERP_TOP, TABLE_INTERP, db, ApiInformation
from datetime import datetime 

FLAG_TABLE_LINK = -3
FLAG_TABLE_HISTORY = 0
LINK_COMMENT = ""

def get_list_pictures(proba=-1):
    base_scr = [
        '../static/pic/inactive1.svg',
        '../static/pic/inactive2.svg',
        '../static/pic/inactive3.svg',
        '../static/pic/inactive4.svg',
        '../static/pic/inactive5.svg',
        ]
    if proba == -1:
        return base_scr
    if proba <= 0.2:
        base_scr[0] = '../static/pic/active1.svg'
    elif proba <= 0.4:
        base_scr[1] = '../static/pic/active2.svg'
    elif proba <= 0.6:
        base_scr[2] = '../static/pic/active3.svg'
    elif proba <= 0.8:
        base_scr[3] = '../static/pic/active4.svg'
    else:
        base_scr[4] = '../static/pic/active5.svg'
    return base_scr

def predict_class(proba, treshold=0.5):
    if proba >= treshold:
        return 1
    return 0

# def update_cell_sql(cell, new_value):
#     cell = new_value


def process_dict_from_proba(dict_data, update_db_proba=True, data_api=None):
    """
        Обработчик входного словаря. Расчет вероятности
    """
    
    comment = None
    #Проверка правильности формирования запроса
    if dict_data == {}:
        comment = "No found required keys: 'question', 'answer'."
    elif "question" not in dict_data.keys():
        comment = "No found required key: 'question."
    elif "answer" not in dict_data.keys():
        comment = "No found required key: 'answer'."
    
    dict_data["log"] = {"comment":comment}
    if comment != None:
        return

    #Проверка api ключа
    if data_api is None:
        data_api = db.session.query(ApiInformation).get(dict_data["key_api"])
        if data_api is None:
            dict_data["log"]["comment"] = "No correct api_key, or unauthorized user"
            return
        elif data_api.counts_proba_left == 0:
            dict_data["log"]["comment"] = "No available query probability"
            return

    question = dict_data["question"]
    answer = dict_data["answer"]
    if "treshold" not in dict_data.keys():
        dict_data["treshold"] = 0.5
    treshold = dict_data["treshold"]

    dict_data["log"]["comment"] = "OK"
    proba = []
    class_ = []
    #Определение максимального числа строк в запросе
    #question список, answer список 
    if isinstance(question,list) and isinstance(answer,list):
        max_len = len(question)
        type_ = 1
    #question строка, answer список
    elif not isinstance(question,list) and isinstance(answer,list):
        max_len = len(answer)
        type_ = 2
    #question список, answer строка
    elif isinstance(question,list) and not isinstance(answer,list):
        max_len = len(question)
        type_ = 3
    #question строка, answer строка
    else:
        max_len = 1
        type_ = 4
    
    if update_db_proba:
        len_seq = min(max_len, data_api.counts_proba_left)
    else:
        len_seq = max_len
    
    if type_ == 1:
        for i, q in enumerate(question[:len_seq]):
            proba.append(MODEL.predict(q, answer[i]))
            class_.append(predict_class(proba[-1], treshold=treshold))
    elif type_ == 2:
        for i, a in enumerate(answer[:len_seq]):
            proba.append(MODEL.predict(question, a))
            class_.append(predict_class(proba[-1], treshold=treshold))
    elif type_ == 3:
        for i, q in enumerate(question[:len_seq]):
            proba.append(MODEL.predict(q, answer))
            class_.append(predict_class(proba[-1], treshold=treshold))
    else:
        proba = MODEL.predict(question, answer)
        class_ = predict_class(proba, treshold=treshold)

    if len_seq != max_len: dict_data["log"]["comment"] = \
            "Some requests were processed. No available query probability"

    dict_data["proba"] = proba
    dict_data["class"] = class_
    
    if update_db_proba:
        #Запись изменений в базу данных
        data_api.counts_proba_left -= len_seq
        db.session.add(data_api)
        db.session.commit()

    dict_data["log"]["count_available_query"] = {"probabylity":data_api.counts_proba_left,
                                                 "link":data_api.counts_link_left,
                                                 "interpretation":data_api.counts_interpr_left
    }
    

def process_dict_from_link(dict_data, update_db_link=True):
    """
        Обработчик входного словаря. Расчет вероятности по ссылке на сайт
    """
    
    if dict_data == {} or ("link" not in dict_data.keys()):
        dict_data["log"] = {"comment":"No found required key: 'link'."}
        return
    dict_data["log"] = {}

    #Проверка api ключа
    data_api = db.session.query(ApiInformation).get(dict_data["key_api"])
    if data_api is None:
        dict_data["log"]["comment"] = "No correct api_key, or unauthorized user"
        return
    elif data_api.counts_link_left == 0:
        dict_data["log"]["comment"] = "No available query probability"
        return

    if "type_parser" not in dict_data.keys():
        dict_data["type_parser"] = "answers_mail.ru"
    
    if dict_data["type_parser"] == "answers_mail.ru":
        data_link = parce_data_from_link_answers_mail_ru(dict_data["link"])
        if data_link["code"] < 0:
            dict_data["log"]["comment"] = data_link["comment"]
            return

    if "treshold" not in dict_data.keys():
        dict_data["treshold"] = 0.5
    treshold = dict_data["treshold"]

    dict_data["question"] = data_link["question"]
    dict_data["answer"] = data_link["answer"]
    dict_data["treshold"] = treshold

    process_dict_from_proba(dict_data, update_db_proba=False, data_api=data_api)
    
    #Запись изменений в базу данных
    data_api.counts_link_left -= 1
    db.session.add(data_api)
    db.session.commit()

    dict_data["log"]["count_available_query"] = {"probabylity":data_api.counts_proba_left,
                                                 "link":data_api.counts_link_left,
                                                 "interpretation":data_api.counts_interpr_left
    }

def process_dict_from_interpretation(dict_data):
    """
        Обработчик входного словаря. Интерпретация
    """
    comment = None
    #Проверка правильности формирования запроса
    if dict_data == {}:
        comment = "No found required keys: 'question', 'answer'."
    elif "question" not in dict_data.keys():
        comment = "No found required key: 'question."
    elif "answer" not in dict_data.keys():
        comment = "No found required key: 'answer'."
    
    dict_data["log"] = {"comment":comment}
    if comment != None:
        return

    dict_data["log"] = {}
    #Проверка api ключа
    data_api = db.session.query(ApiInformation).get(dict_data["key_api"])
    if data_api is None:
        dict_data["log"]["comment"] = "No correct api_key, or unauthorized user"
        return
    elif data_api.counts_interpr_left == 0:
        dict_data["log"]["comment"] = "No available query interpretation"
        return

    neg, pos = interpretation_short(dict_data["question"], dict_data["answer"], dict_data["n_max_top"])

    for key, value in neg.items():
        neg[key] = round(value,3)
    for key, value in pos.items():
        pos[key] = round(value,3)

    dict_data["negative_contribution"] = neg
    dict_data["positive_contribution"] = pos

    #Запись изменений в базу данных
    data_api.counts_interpr_left -= 1
    db.session.add(data_api)
    db.session.commit()

    dict_data["log"]["count_available_query"] = {"probabylity":data_api.counts_proba_left,
                                                 "link":data_api.counts_link_left,
                                                 "interpretation":data_api.counts_interpr_left
    }


def preprocess_link_answers_mail_ru(link_to_site):
    """
        Обработчик входной ссылки на сайт Mail.ru
    """
    if link_to_site is None:
        return None
    elif link_to_site.isdigit():
        return 'https://otvet.mail.ru/question/'+link_to_site
    return link_to_site

def parce_data_from_link_answers_mail_ru(link_to_site):
    output = {}
    output["question"] = None
    output["answer"] = None
    link = preprocess_link_answers_mail_ru(link_to_site)
    if link is None:
        output["code"] = -3
        output["comment"] = ""
        return output
    data = get_question_answers(link)
    if isinstance(data, tuple):
        output["question"] = data[0]
        output["answer"] = data[1]
        if data[1] == []:
            output["code"] = -1
            output["comment"] = "For {} is no answers".format(link)
            return output
        output["code"] = 0
        output["comment"] = "Found question and answers"
    else:
        output["code"] = -2
        output["comment"] = "Url is incorrect format"
        return output
    return output
"""
Main pages
"""
@app_web.route("/", methods=['GET', 'POST']) 
def index():
    return render_template("index.html")

@app_web.route("/q_a/", methods=['GET', 'POST']) 
def q_a():
    global FLAG_TABLE_HISTORY
    result = {}
    result["proba"] = ""
    result["smile"] = get_list_pictures()
    if request.method == "POST":
        FLAG_TABLE_HISTORY = 1
        result["question"] = request.form.get("question")
        result["answer"] = request.form.get("answer")
        result["proba"] = MODEL.predict(result["question"],result["answer"])
        result["smile"] = get_list_pictures(result["proba"])
        TABLE_HISTORY.add_row([result["question"], result["answer"], result["proba"], datetime.today().strftime("%Y-%m-%d_%H.%M.%S")])
        TABLE_HISTORY.save_html(path='app/templates/table_history.html')
    return render_template("index_question_answer.html", result=result)

@app_web.route("/link/", methods=['GET', 'POST']) 
def link():
    global LINK_COMMENT
    global FLAG_TABLE_LINK
    global FLAG_TABLE_HISTORY
    link_to_site = preprocess_link_answers_mail_ru(request.args.get("link_to_site"))
    print(link_to_site)
    data = parce_data_from_link_answers_mail_ru(link_to_site)
    if data["code"] == -3:
        return render_template("index_link.html")
    LINK_COMMENT = data["comment"]
    FLAG_TABLE_LINK = data["code"]

    if data["code"] < 0:
        return render_template("index_link.html")
    FLAG_TABLE_HISTORY = 1
    TABLE.clear_table()
    for answer in data["answer"]:
        proba = MODEL.predict(data["question"], answer)
        TABLE.add_row([data["question"], answer, proba])
        TABLE_HISTORY.add_row([data["question"], answer, proba, datetime.today().strftime("%Y-%m-%d_%H.%M.%S")])
    TABLE.save_html(path='app/templates/table.html')
    TABLE_HISTORY.save_html(path='app/templates/table_history.html')
    return render_template("index_link.html")

@app_web.route("/interpretation/", methods=['GET', 'POST']) 
def interpr():
    result = {}
    result["question"] = ""
    result["answer"] = ""
    if request.method == "POST":
        result["question"] = request.form.get("question")
        result["answer"] = request.form.get("answer")
        interpretation(result["question"], result["answer"])
    return render_template("index_interpretation.html", result=result)

@app_web.route("/service/", methods=['GET', 'POST']) 
def service():
    return render_template("index_service.html")

@app_web.route("/history/", methods=['GET', 'POST']) 
def history():
    return render_template("index_history.html")

"""
Catch Querry
"""
@app_web.route("/catch_query_interpretation/", methods=['GET', 'POST']) 
def catch_interpretation():
    if request.method == "POST":
        data = request.get_json()
        interpretation(data["question"], data["answer"])
    res = make_response("ok")
    return res

@app_web.route("/get_proba/", methods=['POST']) 
def catch_proba():
    data = request.get_json()
    process_dict_from_proba(data)
    return jsonify(data)

@app_web.route("/get_link/", methods=['POST']) 
def catch_link():
    data = request.get_json()
    process_dict_from_link(data)
    return jsonify(data)

@app_web.route("/get_interpretation/", methods=['POST']) 
def get_interpretation():
    data = request.get_json()
    process_dict_from_interpretation(data)
    return jsonify(data)

@app_web.route("/get_user_api_status/", methods=['POST']) 
def query_left():
    data = request.get_json()
    data_api = db.session.query(ApiInformation).get(data["key_api"])
    data["tarif"] = data_api.tarif
    data["counts_proba_left"] = data_api.counts_proba_left
    data["counts_link_left"] = data_api.counts_link_left
    data["counts_interpr_left"] = data_api.counts_interpr_left
    return jsonify(data)

"""
Other Querry
"""
@app_web.route("/output_frame_interpretation/", methods=['GET', 'POST']) 
def frame_interpretation():
    result = {}
    result["question"] = ""
    result["answer"] = ""
    if request.method == "POST":
        result = request.get_json()
    return render_template("form_interpretation.html", result=result)

@app_web.route("/single_plot/", methods=['GET']) 
def single_plot():
    return render_template("single_plot.html")

@app_web.route("/table_top/", methods=['GET']) 
def table_top():
    global TABLE_INTERP_TOP
    return TABLE_INTERP_TOP.get_html(table_id="table_top", class_table="table_top")

@app_web.route("/draw_pos/", methods=['GET']) 
def draw_pos():
    return render_template("Позитивный.html")

@app_web.route("/top_pos/", methods=['GET']) 
def top_pos():
    return render_template("table_interpretation_top_pos.html")

@app_web.route("/top_neg/", methods=['GET']) 
def top_neg():
    return render_template("table_interpretation_top_neg.html")

@app_web.route("/form_interpretation/", methods=['GET', 'POST']) 
def form_interpretation():
    return render_template("single_plot.html")

@app_web.route("/table_proba/", methods=['GET', 'POST']) 
def table_proba():
    global FLAG_TABLE_LINK
    if FLAG_TABLE_LINK < 0:
        return render_template("empty.html", text = LINK_COMMENT)
    return render_template("table.html")
    
@app_web.route("/table_history/", methods=['GET', 'POST']) 
def table_history():
    global FLAG_TABLE_HISTORY
    if not FLAG_TABLE_HISTORY:
        return render_template("empty.html", text = "Ещё нет истории запросов")
    return render_template("table_history.html")

@app_web.route("/table_interpretation/", methods=['GET', 'POST']) 
def table_interpretation():
    global TABLE_INTERP
    return TABLE_INTERP.get_html(table_id="table_interp")