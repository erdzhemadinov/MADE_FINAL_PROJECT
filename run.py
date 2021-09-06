from wapp import app, predict, EXTRACTOR, EMBEDDER

import webbrowser
from threading import Timer

def open_browser():
      webbrowser.open_new_tab("http://localhost:5000/")

# Start Web in default browser
Timer(1, open_browser).start()
# Model warmup. Make first predict
predict(EMBEDDER, EXTRACTOR, question="Вопрос", answer="Ответ")

app.run(debug=False)
