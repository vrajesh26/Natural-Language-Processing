from flask import Flask,render_template,url_for,request,redirect
import pandas as pd
from werkzeug import secure_filename
from nlp_model import sentiment_final
import os
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/sample1",methods=['GET','POST'])
def sample1():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        if filename.endswith(".csv"):
            input_data_tweet = pd.read_csv(file)
            col_name = request.form["col_name"]
            if col_name in input_data_tweet.columns:
                #input_data_tweet.rename({col_name:'tweet'},axis=1,inplace=True)
                t=sentiment_final(input_data_tweet[col_name],name=col_name)
                x=t.cloud()
                disp="Most Frequent Words"
                return render_template('sample1.html',user_image = x,disp=disp)
            else:
                invalid_col="Invalid column name"
                return render_template('sample1.html',invalid=invalid_col)
        elif filename.endswith(".xlsx"):
            input_data_tweet = pd.read_excel(file)
            col_name = request.form["col_name"]
            if col_name in input_data_tweet.columns:
                #input_data_tweet.rename({col_name:'tweet'},axis=1,inplace=True)
                t=sentiment_final(input_data_tweet[col_name],name=col_name)
                x=t.cloud()
                disp="Most Frequent Words"
                return render_template('sample1.html',user_image = x,disp=disp)
            else:
                invalid_col="Invalid column name"
                return render_template('sample1.html',invalid=invalid_col) 
        else:
            formating="File format not supported"
            return render_template('sample1.html',formating=formating)
    return render_template('sample1.html')

@app.route("/sample2",methods=['GET','POST'])
def sample2():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        if filename.endswith(".csv"):
            df = pd.read_csv(file)
            disp="Number of tweets:"
            return render_template('sample2.html', shape=df.shape[0],tables=df.to_html(classes='table table-striped table-hover'),disp=disp)
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(file)
            disp="Number of tweets:"
            return render_template('sample2.html', shape=df.shape[0],tables=df.to_html(classes='table table-striped table-hover'),disp=disp)
        else:
            formating="File format not supported"
            return render_template('sample2.html',formating=formating)
    return render_template('sample2.html')

@app.route("/sample3",methods=['GET','POST'])
def sample3():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        if filename.endswith(".csv"):
            df = pd.read_csv(file)
            col_name = request.form["col_name"]
            if col_name in df.columns:
                #df.rename({col_name:'tweet'},axis=1,inplace=True)
                t=sentiment_final(df[col_name],name=col_name)
                x=t.analyse_1(col_name)
                return render_template('sample3.html',tables=x.to_html(classes='table table-striped table-hover'))
            else:
                invalid_col="Invalid column name"
                return render_template('sample3.html',invalid=invalid_col)
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(file)
            col_name = request.form["col_name"]
            if col_name in df.columns:
                #df.rename({col_name:'tweet'},axis=1,inplace=True)
                t=sentiment_final(df[col_name],name=col_name)
                x=t.analyse_1(col_name)
                return render_template('sample3.html',tables=x.to_html(classes='table table-striped table-hover'))
            else:
                invalid_col="Invalid column name"
                return render_template('sample3.html',invalid=invalid_col)
        else:
            formating="File format not supported"
            return render_template('sample3.html',formating=formating)
        
    return render_template('sample3.html')

@app.route("/sample4",methods=['GET','POST'])
def sample4():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        if filename.endswith(".csv"):
            df = pd.read_csv(file)
            col_name = request.form["col_name"]
            if col_name in df.columns:
                #df.rename({col_name:'tweet'},axis=1,inplace=True)
                t=sentiment_final(df[col_name],name=col_name)
                x=t.sentiment_score()
                data=x[[col_name,'Sentiment']]
                return render_template('sample4.html', tables=data.to_html())
            else:
                invalid_col="Invalid column name"
                return render_template('sample4.html',invalid=invalid_col)
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(file)
            col_name = request.form["col_name"]
            if col_name in df.columns:
                #df.rename({col_name:'tweet'},axis=1,inplace=True)
                t=sentiment_final(df[col_name],name=col_name)
                x=t.sentiment_score()
                data=x[[col_name,'Sentiment']]
                return render_template('sample4.html', tables=data.to_html())
            else:
                invalid_col="Invalid column name"
                return render_template('sample4.html',invalid=invalid_col)
        else:
            formating="File format not supported"
            return render_template('sample4.html',formating=formating)
    return render_template('sample4.html')

@app.route("/sample5",methods=['GET','POST'])
def sample5():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        if filename.endswith(".csv"):
            df = pd.read_csv(request.files.get('file'))
            Num_of_topics = request.form["Num_of_topics"]
            col_name = request.form["col_name"]
            if col_name in df.columns:
                #df.rename({col_name:'tweet'},axis=1,inplace=True)
                t=sentiment_final(df[col_name],name=col_name)
                dat=t.top_model(number_of_topics=Num_of_topics)
                disp="Most Discussed Topics"
                y=[]
                for i in dat:
                    y.append(i[1])
                return render_template('sample5.html', topics=y, disp=disp)
            else:
                invalid_col="Invalid column name"
                return render_template('sample5.html',invalid=invalid_col)
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(file)
            Num_of_topics = request.form["Num_of_topics"]
            col_name = request.form["col_name"]
            if col_name in df.columns:
                #df.rename({col_name:'tweet'},axis=1,inplace=True)
                t=sentiment_final(df[col_name],name=col_name)
                dat=t.top_model(number_of_topics=Num_of_topics)
                disp="Most Discussed Topics"
                y=[]
                for i in dat:
                    y.append(i[1])
                return render_template('sample5.html', topics=y, disp=disp)
            else:
                invalid_col="Invalid column name"
                return render_template('sample5.html',invalid=invalid_col)
        else:
            formating="File format not supported"
            return render_template('sample5.html',formating=formating)
    return render_template('sample5.html')

if __name__ == "__main__":
    app.run(debug=True)
