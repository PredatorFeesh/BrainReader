import os
from flask import Flask

#The seperate websites import
from web import web
from pred import api
from databaseModel import database


app = Flask(__name__)
app.register_blueprint(api, url_prefix='/api')
app.register_blueprint(database, url_prefix='/database')
app.register_blueprint(web)

# Load default config and override config from an environment variable
app.config.update(dict(
    SECRET_KEY=r'"yf*!c2slj-s(bt493g@&-a$cr8+-l&niez6)-x2d$%#sx%$12s"',
    UPLOAD_FOLDER=os.path.dirname(os.path.realpath(__file__)) + "\\csv_files"
))

app.config.from_envvar('FLASKR_SETTINGS', silent=True)

if __name__ == '__main__':
    app.run(debug=False)