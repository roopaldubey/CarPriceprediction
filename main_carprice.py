from flask import Flask,redirect,url_for,render_template,session,jsonify,request;
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import numpy as np
import sklearn.tree
import pickle

app = Flask(__name__);

app.secret_key = 'secret key'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'roopal'
app.config['MYSQL_DB'] = 'pythonregister'

mysql = MySQL(app);
model = pickle.load(open('finalized_model.sav', 'rb'))

@app.route("/",methods=['POST','GET'])
def loginblock():
    msg=' '
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
		
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password,))
        account = cursor.fetchone()
        if account:
            session['loggedin'] = True
            session['username'] = account['username']
            msg='Welcome '+username;
            return render_template('loginresult.html',msg=msg);
        else:
            msg = 'Incorrect username/password!'
    return render_template('websitehomepage.html', msg=msg)

        
@app.route("/register/",methods=['POST','GET'])
def registerpage():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'email' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO accounts VALUES (%s, %s, %s)', (username, email, password,))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('registration.html', msg=msg)
@app.route("/loginresult/DecisionTree/",methods=['POST','GET'])
def predictDtree():

    if request.method == 'POST':

        form_ftrs  =  [x for x in request.form.values()]
        end_ftrs = [np.array(form_ftrs)]
        data_prediction = model.predict(end_ftrs)
        output = round(data_prediction[0], 2)
        return render_template('DecisionTree.html', prediction_text='The new price is Rs {}'.format(output))
    else:
        return render_template('DecisionTree.html',prediction_text='You should choose correct output')

@app.route("/loginresult/RandomForest/",methods=['POST','GET'])
def predictRforest():

    if request.method == 'POST':

        form_ftrs  =  [x for x in request.form.values()]
        end_ftrs = [np.array(form_ftrs)]
        data_prediction = model.predict(end_ftrs)
        output = round(data_prediction[0], 2)
        return render_template('RandomForest.html', prediction_text='The new price is Rs {}'.format(output))
    else:
        return render_template('RandomForest.html',prediction_text='You should choose correct output')
@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
    
if __name__=='__main__':
	app.run(debug=True);
