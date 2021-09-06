from flask_wtf import FlaskForm
from wtforms import TextField
from wtforms.validators import Required


class QAForm(FlaskForm):
    question = TextField('question', validators = [Required()])
    answer = TextField('answer', validators = [Required()])

class UrlForm(FlaskForm):
    url = TextField('url', validators = [Required()])
