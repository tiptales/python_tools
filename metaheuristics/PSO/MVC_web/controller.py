
from flask import Flask, render_template, request
from compute import compute
from model import InputForm

app = Flask(__name__)

@app.route('/MVC_web', methods=['GET', 'POST'])
def index():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        p = form.p.data
        t = form.t.data
        s = compute(p, t)
    else:
        s = None

    return render_template("view.html", form=form, s=s)


if __name__ == '__main__':
    app.run(debug=True)
