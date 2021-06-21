import pandas as pd
from sklearn.model_selection import train_test_split
import re
import string
from flask import Flask,render_template,request
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


app=Flask(__name__,instance_path=r"C:\Users\admin\Documents\projects\fake news detection\app.py")
news=pd.read_csv("merged_df.csv")
x=news["text"]
y=news["class"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)

vectorization= TfidfVectorizer()
xv_train=vectorization.fit_transform(x_train)
xv_test=vectorization.transform(x_test)

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()
dt.fit(xv_train,y_train)

dt.score(xv_test,y_test)
pickle.dump(dt,open("model.pkl",'wb'))
def word_drop(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

model=pickle.load(open("model.pkl",'rb'))


@app.route('/')
def hello_world():
	return render_template("fake_news.html")


@app.route("/predict",methods=['POST','GET'])
def predict():
    features=[x for x in request.form.values()]
    news=str(features[0])
    testing_news={'text':[news]}
    new_def_test=pd.DataFrame(testing_news)
    new_def_test["text"]=new_def_test["text"].apply(word_drop)
    new_x_test=new_def_test["text"]
    new_xv_test=vectorization.transform(new_x_test)
    pred_dt=model.predict(new_xv_test)
    if pred_dt==0:
        return render_template('fake_news.html',pred="Fake News") 
    elif pred_dt==1:
        return render_template('fake_news.html',pred="Not a Fake News")
    
if __name__ == "__main__":
	app.run(debug=True)
        