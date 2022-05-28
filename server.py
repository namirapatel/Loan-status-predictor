from flask import Flask
from flask import request,render_template
app = Flask(__name__)
from test import loan_predict


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/loan',methods = ['POST'])
def loan():
    data = request.form # a multidict containing POST data
    l=[]
    print("*********************")
    print(dict(data))
    for key,value in dict(data).items():
        l.append(value)
    output=str(loan_predict(l))

    return render_template('index.html',prediction_text ="the loan status is {}".format(output))
    # return 'Hello, World!'

if __name__ == '__main__':
    app.run()