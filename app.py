from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/check", methods=["POST"])
def check():
    neighbourhood_group = request.form["neighbourhood_group"]
    room_type = request.form["room_type"]
    number_of_reviews = request.form["number_of_reviews"]
    calculated_host_listings_count = request.form["calculated_host_listings_count"]
    d = [[neighbourhood_group, room_type, number_of_reviews, calculated_host_listings_count]]
    with open("rf.model", "rb") as f:
        model = pickle.load(f)
    result = model.predict(d)
    return render_template("index.html", msg=result)

# @app.route("/display")

if __name__  =="__main__":
    app.run(debug=True)

