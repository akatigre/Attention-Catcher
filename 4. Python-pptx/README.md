# Python PPTX
![slide1](images/slide1.PNG)
```python
from pptx import Presentation
from pptx.util import Inches
from pptx.enum.shapes import MSO_SHAPE
from pptx_tools import utils


with open("summerization.txt", 'r', encoding="UTF-8") as file:
    string = file.readlines()
with open("relation.txt", 'r', encoding="UTF-8") as file:
    relation = file.read()
relation = [i for i in relation.split('\n\n') if i]


prs = Presentation("template1.pptx") 


slide_1 = prs.slides[0] # slide1

slide_1.placeholders.element[3][2][2][1][1].text = string[1]
slide_1.placeholders.element[4][2][2][1][1].text = string[2]
slide_1.placeholders.element[5][2][2][1][1].text = string[3]

slide_1.placeholders.element[9][2][2][1][1].text = relation[1][0]
slide_1.placeholders.element[18][2][2][1][1].text = relation[1][1]
slide_1.placeholders.element[13][2][2][1][1].text = relation[1][2]
slide_1.placeholders.element[19][2][2][1][1].text = relation[1][3]
slide_1.placeholders.element[19][2][2][1][1].text = relation[1][4]


slide_1.placeholders.element[21][2][2][1][1].text = relation[2][0]
slide_1.placeholders.element[22][2][2][1][1].text = relation[2][1]
slide_1.placeholders.element[23][2][2][1][1].text = relation[3][2]
slide_1.placeholders.element[38][2][2][1][1].text = relation[1][3]
slide_1.placeholders.element[38][2][2][1][1].text = relation[1][4]

slide_2 = prs.slides[1]
###########################################################################################################
                                        <  ...  >
##########################################################################################################
prs.save('test1.pptx') 
utils.save_pptx_as_png("C:\\Users\\Erin Lee\\Desktop\\new", 'C:\\Users\Erin Lee\PycharmProjects\ACatcher\slides.pptx')
```
