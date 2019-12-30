
from wtforms import Form, StringField, FloatField, IntegerField, validators

class InputForm(Form):
    p = StringField(validators=[validators.InputRequired()])
    t = FloatField(validators=[validators.InputRequired()])
    #length = IntegerField(validators=[validators.InputRequired()])
#p = None    # input  former r
#s = None   # output former s