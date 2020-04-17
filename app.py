from flask import Flask, render_template, request, redirect, url_for
from ML import PredictDisease
import json
import socket

hostname = socket.gethostname()
ip_addr = socket.gethostbyname(hostname)

app = Flask(__name__)
data_model = PredictDisease()

app.my_ip = ip_addr
app.my_port = 80


@app.route('/predict/<location>', methods=['POST'])
def predict(location):
    if request.method == 'POST':
        if location == 'first_list':
            return json.dumps({'result': data_model.get_disease_list()})
        if location == 'first_iter':
            d_name = request.form['d_name']
            if len(d_name) == 0:
                return json.dumps({'final': -1})
            else:
                return json.dumps(data_model.predict(d_name))
        if location == 'second_iter':
            node = request.form['node']
            decision = request.form['decision']
            if decision == 'true':
                decision = 1
            elif decision == 'false':
                decision = 0
            return json.dumps(data_model.recurse_predict(node, decision))
        if location == 'final_result':
            node = request.form['node']
            d_list = json.loads(request.form['d_list'])
            return json.dumps(data_model.final_result(node, d_list))


@app.route('/')
def root():
    return redirect(url_for('home'))


@app.route('/disease_p')
def disease_p():
    return render_template('disease_p.html')


@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/blog')
def blog():
    return render_template('blog.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/dep')
def dep():
    return render_template('dep.html')


@app.route('/doctor')
def doctor():
    return render_template('doctor.html')


@app.route('/elements')
def elements():
    return render_template('elements.html')


@app.route('/services')
def services():
    return render_template('services.html')


@app.route('/single_blog')
def single_blog():
    return render_template('single-blog.html')


if __name__ == '__main__':
    app.run(host=app.my_ip, port=app.my_port)
