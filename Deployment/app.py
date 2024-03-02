from flask import Flask, render_template,request
import pandas as pd
import numpy as np
app = Flask(__name__)
import pickle
model = pickle.load(open("final.pkl",'rb'))
'''def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)'''

#model = pickle.load(open("final.pkl",'rb'))
@app.route('/')
def home():
    return render_template("index.html")
@app.route('/guest', methods =["post"])
def Guest():
    rc = request.form["rc"]
    fc = request.form["fc"] 
    mn = request.form["mn"]
    
    def norm_func(i):
        x = (i-i.min())	/ (i.max()-i.min())
        return (x)    
    data  = [[rc,fc,mn]]
    data1 = np.array(data, dtype=float)
    data2 = norm_func(data1)                
    output = model.predict(data2)
    
    #prediction = prediction[0][0]
    return render_template("index.html",y = "result is" + ' ' + format(output))
if __name__ == '__main__':
    app.run(debug = True)