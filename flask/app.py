from flask import Flask, render_template, request
from flask_admin.contrib.fileadmin import FileAdmin
from flask_admin import Admin
from flask_dropzone import Dropzone  # drop box
import os

app = Flask(__name__)
app.debug = True

basedir = os.path.abspath(os.path.dirname(__file__))  # 현재 파일의 dir 절대경로, python console 에 쓰면 동일 결과
upload_dir = os.path.join(basedir, 'uploads')  # basedir 의 uploads 에 파일 저장
#################################################################################################################

admin = Admin(name='Uploaded Files')
admin.init_app(app)  # 이제 실행하고 주소창에 /admin 하면 창 나옴
dropzone = Dropzone(app)
admin.add_view(FileAdmin(upload_dir, name='FILES'))  # /admin 가면 올린 파일 관리 가능
#################################################################################################################


@app.route("/", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':  # print(request.files) 하면 dic 인 것을 알 수 있음
        f = request.files.get('file')  # get(for dictionary) doesn't create error
        f.save(os.path.join(upload_dir, f.filename))
    return render_template('homepage.html')

##############################################################################


if __name__ == '__main__':
    app.run()
