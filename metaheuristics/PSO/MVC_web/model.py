
from wtforms import Form, StringField, FloatField, IntegerField, validators

class InputForm(Form):
    n_iterations = IntegerField(validators=[validators.InputRequired()])
    target_error = FloatField(validators=[validators.InputRequired()])
    n_particles = IntegerField(validators=[validators.InputRequired()])
    #length = IntegerField(validators=[validators.InputRequired()])

