import os
from multiprocessing import Pool
from werkzeug.utils import secure_filename
from flask import request, redirect, url_for, render_template, Blueprint

import databaseModel

ALLOWED_EXTENSIONS = set(['csv'])
UPLOAD_FOLDER=os.path.dirname(os.path.realpath(__file__)) + "\\csv_files"

web = Blueprint('web', __name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@web.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            os.rename(os.path.join(UPLOAD_FOLDER, filename),
                      os.path.join(UPLOAD_FOLDER) + r'\\' + request.form["fname"] + ".csv")
            pool = Pool(processes=2)
            print("I'm not here yet")
            if "catOrTest" in request.form:
                pool.apply_async(databaseModel.write_to_db, (request.form["cat"], request.form["fname"], True, request.form['posneg'],))
            else:
                pool.apply_async(databaseModel.write_to_db, (request.form["cat"], request.form["fname"], False, request.form['posneg'],))
            return redirect(url_for('web.upload_file'))
    return render_template('index.html')
