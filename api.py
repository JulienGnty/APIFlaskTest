from flask import Flask, jsonify, request, render_template
import joblib

app = Flask(__name__, template_folder='templates')

# 0 - hate speech 1 - offensive language 2 - neither

Labels = ["Hate speech", "Offensive language", "Neither"]
clf = joblib.load("model/hatespeech.joblib.z")

@app.route('/')
def hello():
    return render_template('index.html')

# Route pour créer un nouvel étudiant
@app.route('/prediction/<string:texte>', methods=['GET'])
def predict_word(texte):
    return jsonify(Labels[clf.predict_proba([texte]).argmax()])

# Route pour créer un nouvel étudiant
@app.route('/prediction', methods=['POST'])
def predict_text():
	return render_template('index.html', response = Labels[clf.predict_proba([request.form['texte']]).argmax()])
    #return jsonify(Labels[clf.predict_proba([request.json.get('texte')]).argmax()])

# Lancement de l'application Flask
if __name__ == '__main__':
    app.run(debug=True)

