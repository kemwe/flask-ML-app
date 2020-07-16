from flask import Flask,jsonify,request
#from sklearn.externals import joblib
import pandas as pd
#import joblib
#import numpy as np
import traceback
import pickle


app=Flask(__name__)
#lr = joblib.load("model.pkl")  # Load "model.pkl"
lr = pickle.load(open('model2.pkl','rb'))
print('Model loaded')
model_columns = pickle.load(open("model_columns.pkl","rb"))  # Load "model_columns.pkl"
print('Model columns loaded')

@app.route("/predict", methods=["POST"])
def predict():
    if lr:
        try:

            json_=request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            prediction = list(lr.predict(query))

            return jsonify({'prediction':str(prediction)})

        except:

            return jsonify({'trace':traceback.format_exc()})

    else:
        print('Train the model first')
        return ('No model here to use')


if __name__ == '__main__':
    try:
        port = 6001  # This is for a command-line input
    except:
        port = 12345  # If you don't provide any port the port will be set to 12345

    

    app.run(port=port, debug=True)