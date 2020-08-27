from flask import Flask, render_template, request
from flask_admin.contrib.fileadmin import FileAdmin
from flask_admin import Admin
from flask_dropzone import Dropzone  # drop box
import os
from pdf2image import convert_from_path

app = Flask(__name__)
app.debug = True

basedir = os.path.abspath(os.path.dirname(__file__))  # 현재 파일의 dir 절대경로, python console 에 쓰면 동일 결과
upload_dir = os.path.join(basedir, 'uploads')  # basedir 의 uploads 에 파일 저장
#################################################################################################################

admin = Admin(name='Uploaded Files')
admin.init_app(app)  # 이제 실행하고 주소창에 /admin 하면 창 나옴
dropzone = Dropzone(app)
admin.add_view(FileAdmin(upload_dir, name='FILES'))  # /admin 가면 올린 파일 관리 가능
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*, .pdf, .txt'
#################################################################################################################


@app.route("/", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        f.save(os.path.join(upload_dir, f.filename))
        images = convert_from_path(os.path.join(upload_dir, f.filename), poppler_path=r"poppler\bin")
        for i, image in enumerate(images):
            fname = "uploads/image" + str(i) + ".jpg"
            image.save(fname, "JPEG")
    return render_template('homepage.html')
##############################################################################


@app.route("/result", methods=['GET', 'POST'])
def upload2():
    if request.method == 'POST':
        f = request.files.get('file')
        f.save(os.path.join(upload_dir, f.filename))
    return render_template('result.html')


# def main():
#     txt = ocr(text.jpeg)
#     html=highlight.main(txt)
#     processed = process(html)
#     return processed
# f.save(os.path.join(upload_dir, f.filename))

###############################################################################
if __name__ == '__main__':
    app.run()
