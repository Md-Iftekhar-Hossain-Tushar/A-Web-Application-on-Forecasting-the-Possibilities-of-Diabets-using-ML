from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the pickled model
with open('clf.pkl', 'rb') as clf_file:
    clf = pickle.load(clf_file)


    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_diabates', methods=['POST'])
def predict_Diabates(): 
        if request.method =='POST':
            Pregnancies = int(request.form.get('Pregnancies'))
            Glucose = int(request.form.get('Glucose'))
            BloodPressure = int(request.form.get('BloodPressure'))
            SkinThickness = int(request.form.get('SkinThickness'))
            Insulin = int(request.form.get('Insulin'))
            BMI = float(request.form.get('BMI'))
            DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
            Age = int(request.form.get('Age'))

        prediction_arr = clf.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
       
        # Return the prediction result
        if prediction_arr[0]==1:
            diabates_Prediction = "High Possibilities"
        else:
            diabates_Prediction = "Less Possibilities"
        return render_template('predict.html', diabetes_Prediction=diabates_Prediction)



   
if __name__ == '__main__':
    app.run(debug=True)