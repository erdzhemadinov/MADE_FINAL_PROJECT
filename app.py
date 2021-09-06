from app import app_web, manager, MODEL

import webbrowser
from threading import Timer
from os import remove

def open_browser():
      webbrowser.open_new_tab("http://localhost:5000/")

START_BROWSER = True

if __name__ == "__main__":
      if START_BROWSER:
            Timer(1, open_browser).start()
      # Model warmup (ускоряет первый запуск)
      MODEL.predict(question="Вопрос", answer="Ответ")
      # Запуск базового сервера
      app_web.run(host="0.0.0.0")
      #manager.run()
      """
      manager.run()
      #Запуск сервера, через консоль
      #python app.py runserver --host 127.0.0.1 --port 5000
      """