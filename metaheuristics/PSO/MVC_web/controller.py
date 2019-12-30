
from flask import Flask, render_template, request
from compute import compute
from model import InputForm

app = Flask(__name__)


@app.route('/MVC_web', methods=['GET', 'POST'])
def index():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        n_iterations = form.n_iterations.data
        target_error = form.target_error.data
        n_particles = form.n_particles.data
        s_1, s_2, i = compute(n_iterations, target_error, n_particles)


    else:
        s_1 = None
        s_2 = None
        i = None


    return render_template("view.html", form=form, s_1=s_1, s_2=s_2, i=i)


if __name__ == '__main__':
    app.run(debug=True)
