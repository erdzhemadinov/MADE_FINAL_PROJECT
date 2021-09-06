from __future__ import unicode_literals
from flask import Flask, request, render_template

from bertolet import load_embedder, predict
import webbrowser
from threading import Timer
from argparse import ArgumentParser

app = Flask(__name__)


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--cpu", "-c", help="Variant of calculation",
                        default=False, type=bool)
    return parser.parse_args()


def open_browser():
      webbrowser.open_new_tab("http://localhost:5000/")


@app.route("/", methods=['GET', 'POST'])
def index():
    result = {}
    result["question"] = ""
    result["answer"] = ""
    result["proba"] = ""
    if request.method == "POST":
        result["question"] = request.form.get("question")
        result["answer"] = request.form.get("answer")
        result["proba"] = predict(embbeder, result["question"], result["answer"])
        result["proba"] = round(result["proba"][0][1], 5)
    return render_template("template.html", result=result)


if __name__ == "__main__":
    args = parse_arguments()
    # Model warmup (ускоряет первый запуск)
    embbeder = load_embedder()
    predict(embbeder, question="Вопрос", answer="Ответ")
    # Открытие страницы браузера через 1 секунду
    Timer(1, open_browser).start();
    # Запуск Web
    app.run()
