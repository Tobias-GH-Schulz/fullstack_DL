import requests
import datetime as dt
from uuid import uuid4
import sqlite3
from pathlib import Path
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


from flask import Flask, jsonify, render_template, request, url_for, redirect


template_dir = Path("../templates")
app = Flask(__name__, template_folder=str(template_dir))


def verify_hash(plain_password, hashed_password):
    """
    This functions returns True if the password matches the hash,
    otherwise it returns False
    """

    return pwd_context.verify(plain_password, hashed_password)


def get_hash(password):
    return pwd_context.hash(password)


class DB:
    def __init__(self, dbname):
        self.dbname = dbname
        self.conn = sqlite3.connect(dbname, check_same_thread=False)

        with self.conn as c:
            c.executescript(
                """
CREATE TABLE IF NOT EXISTS logs (time TEXT, key TEXT, value TEXT);
CREATE TABLE IF NOT EXISTS users (user_id TEXT, name TEXT, email TEXT, password TEXT);
""".strip()
            )

    def create_user(self, name, email, password):

        hashed_password = get_hash(password)
        new_user_id = str(uuid4())
        with self.conn as c:
            c.execute(
                "insert into users values (?, ?, ?, ?)",
                (new_user_id, name, email, hashed_password),
            )

        return new_user_id

    def check_user(self, name):

        with self.conn as c:
            [exists] = c.execute('SELECT EXISTS(SELECT 1 FROM users WHERE name = ?)', (name,)).fetchone()
            if [exists] == [1]:
                return True
            else:
                return False

    def validate_password(self, email, password):
        """This function receives an email and password and checks
        if that's the password associated with that email.

        If the they don't match it returns None, if they match
        it will return the user_id associated with that user.
        """

        user = self.conn.execute(
            "select * from users where email = ?", (email,)
        ).fetchone()

        print(user)

        if not user:
            return None
        else:
            user_id = user[0]
            name = user[1]
            email = user[2]
            hashed_password = user[3]

            print(password, hashed_password)

            if not verify_hash(password, hashed_password):
                return None
            else:
                return user_id

    def log_message(self, key, value):

        now = dt.datetime.utcnow().isoformat()

        with self.conn as c:
            c.execute("INSERT INTO logs VALUES (?, ?, ?)", (now, key, value))

        return

db = DB(dbname="ml_app.db")

app = Flask(__name__)

@app.route("/", methods=["GET"])
@app.route("/index", methods=["GET"])
def home():

    return render_template("/index.html")

@app.route("/login", methods=["POST", "GET"])
def login():
    return render_template("/login.html")

@app.route("/sign_up", methods=["POST", "GET"])
def sign_up():
    return render_template("/sign_up.html")

@app.route("/validate_login", methods=["POST"])
def validate_login():
    if request.method == "POST":
        email = request.form["Email"]
        password = request.form["Password"]
        print(hashed_pass)
        if db.validate_password(email, password):
            #return redirect(url_for('logged'))
            return "password accepted"
        else:
            return "password not accepted"
            #return redirect(url_for("login"))

@app.route("/create_user", methods=["POST"])
def new_user():
    if request.method == "POST":
        name = request.form["Name"]
        email = request.form["Email"]
        hashed_pass = get_hash(request.form["Password"])
        if not db.check_user(name):
            db.create_user(name, email, hashed_pass)
            return redirect(url_for('logged'))
        else:
            return "user already exists"
        
@app.route("/logged", methods=["POST", "GET"])
def logged():

    return render_template("/upload.html")



@app.route("/submit", methods=["POST"])
def request_predict():

    if request.method == "POST":
        password = request.form["Password"]
        email = request.form["Email"]
        if not db.validate_password(email=email, password=password):
            return "not allowed"
            # better error needed
        file = request.files["file"]
        img_bytes = file.read()
        
        r = requests.post("http://127.0.0.1:5000/predict", files={"file": img_bytes})

        r.raise_for_status()

        result_class = r.json()

        return render_template("/response.html", response=result_class)


