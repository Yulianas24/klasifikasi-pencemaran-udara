from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def klasifikasi_udara():
    if request.method == 'GET':
        return render_template("main_page.html")
    elif request.method == 'POST':
        print(dict(request.form))
        features = dict(request.form).values()
        features = np.array([float(x) for x in features])
        model, std_scaler = joblib.load(
            "model-development\klasifikasi_pencemaran_udara.pkl")
        features = std_scaler.transform([features])
        print(features)
        result = model.predict(features)
        return render_template('main_page.html', result=result)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)
