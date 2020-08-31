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


with open("result.txt", 'r', encoding="UTF-8") as file:
    string = file.readlines()
with open("relation.txt", 'r', encoding="UTF-8") as file:
    relation = file.read()
relation = [i for i in relation.split('\n\n') if i]
with open("templates/result_1.html", "r", encoding="UTF-8") as file:
    result_1 = file.read()
with open("templates/result_1_1.html", "r", encoding="UTF-8") as file:
    result_1_1 = file.read()
with open("templates/result_1_2.html", "r", encoding="UTF-8") as file:
    result_1_2 = file.read()
with open("templates/result_1_3.html", "r", encoding="UTF-8") as file:
    result_1_3 = file.read()
with open("templates/result_1_4.html", "r", encoding="UTF-8") as file:
    result_1_4 = file.read()
with open("templates/result_1_5.html", "r", encoding="UTF-8") as file:
    result_1_5 = file.read()
# result_1 += ("<div class='col-md-7'<hr><div class='row-eq-height'><div class='col-md-12' style='line-height: 40px'><p class='dark-grey-text'><mark><strong>Paragraph" + str(1) + "</strong></mark></p>")
# print(result_1)
#for i in range(len(string)):


for i in range(1, len(string)):
    result_1 += ("<div class='col-md-7'><hr><div class='row-eq-height'><div class='col-md-12' style='line-height: 40px'><p class='dark-grey-text'><mark><strong>Paragraph " + str(i) + "</strong></mark></p><a style='line-height: 40px'>")
    result_1 += string[i] # paragraph
    result_1 += result_1_1
    result_1 += ("<img src='" + "https://raw.githubusercontent.com/yoonkim313/dataCampusProject-Team10/master/flask/exslide1.png" + "' class='img-fluid' alt='Sample post image'>")  # ppt slide picture
    # result_1 += ("<img src='" + picture[i+1] + "' class='img-fluid' alt='Sample post image'>")
    result_1 += result_1_2
    result_1 += ("<button type='button' class='btn-outline-deep-orange' data-toggle='modal' data-target='#P" + str(i) + "'>Relation</button><div class='modal' id='P" + str(i) + "'>")  # relation tagging modal
    result_1 += result_1_3
    result_1 += relation[i-1]  # relation tagging input
    result_1 += result_1_4
result = result_1 + result_1_5
with open("templates/final_result.html", "w", encoding="UTF-8") as file:
    file.write(result)


@app.route("/result", methods=['GET', 'POST'])
def upload2():
    if request.method == 'POST':
        f = request.files.get('file')
        f.save(os.path.join(upload_dir, f.filename))
    return render_template('final_result.html')


# def main():
#     txt = ocr(text.jpeg)
#     html=highlight.main(txt)
#     processed = process(html)
#     return processed
# f.save(os.path.join(upload_dir, f.filename))

###############################################################################
if __name__ == '__main__':
    app.run()
