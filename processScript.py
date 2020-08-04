import requests
import re

class processScript():
    def __init__(self,object):
        self.object = object
        self.alphabets = "([A-Za-z])"
        self.prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
        self.suffixes = "(Inc|Ltd|Jr|Sr|Co)"
        self.starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
        self.acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
        self.websites = "[.](com|net|org|io|gov)"

    def summarize(self):
        object = self.object.replace('\n', '')
        object = object.replace('\t', '')
        object = object.replace(' ', '')
        object = object.replace('"', '')
        url = "https://api.smrzr.io/summarize?ratio=0.15"
        headers = {
            'content-type': 'raw/text',
            'origin': 'https://smrzr.io',
            'referer': 'https://smrzr.io/',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
            "user-agent": "Mozilla/5.0"
        }
        resp = requests.post(url, headers=headers, data=object.encode('utf-8'))
        assert resp.status_code == 200

        summary = resp.json()['summary']
        a = summary.split('\n')

        return a



    def split_into_sentences(self,text):
        text = " " + text + "  "
        text = text.replace("\n", " ")
        text = re.sub(self.prefixes, "\\1<prd>", text)
        text = re.sub(self.websites, "<prd>\\1", text)
        if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
        text = re.sub("\s" + self.alphabets + "[.] ", " \\1<prd> ", text)
        text = re.sub(self.acronyms + " " + self.starters, "\\1<stop> \\2", text)
        text = re.sub(self.alphabets + "[.]" + self.alphabets + "[.]" + self.alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
        text = re.sub(self.alphabets + "[.]" + self.alphabets + "[.]", "\\1<prd>\\2<prd>", text)
        text = re.sub(" " + self.suffixes + "[.] " + self.starters, " \\1<stop> \\2", text)
        text = re.sub(" " + self.suffixes + "[.]", " \\1<prd>", text)
        text = re.sub(" " + self.alphabets + "[.]", " \\1<prd>", text)
        if "”" in text: text = text.replace(".”", "”.")
        if "\"" in text: text = text.replace(".\"", "\".")
        if "!" in text: text = text.replace("!\"", "\"!")
        if "?" in text: text = text.replace("?\"", "\"?")
        text = text.replace(".", ".<stop>")
        text = text.replace("?", "?<stop>")
        text = text.replace("!", "!<stop>")
        text = text.replace("<prd>", ".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]
        return sentences



    def main(self):
        summarized = self.summarize()[0]
        sentences = self.split_into_sentences(summarized)
        return sentences
        


txt1 ='''
Tired of making PowerPoints? This is for you! 안녕하세요 TEXT2PPTX라는 새로운 서비스를 소개하고자 이 자리에 서게 되었습니다. 텍스트 투 피피티라는 서비스는 사용자가 대본을 입력하면 자동으로 파워포인트를 생성해주는 서비스입니다. PPTX를 만드는 데에 소요되는 시간을 아끼고 싶으신가요? 디자인과 내용 면에서 양질의 피피티를 만들고 싶으신가요? 여러분들에게 발표의 새로운 패러다임을 열어줄 서비스를 지금 소개해 드리겠습니다.
'''
txt2='''
텍스트 투 피피티에서 제공하는 서비스는 크게 세 가지가 있습니다. 첫째, 대본을 자동으로 요약하고 중심 내용을 추출하여 적재적소에 전달하고자하는 내용을 세련된 방식으로 피피티로 만들어드립니다. 두번째, 피피티를 디자인하는 것에 어려움을 겪고 있는 사용자를 위해 폰트, 템플릿, 벡터 이미지 등을 보기쉬운 인터페이스로 추천해드립니다. 추천 시스템에 의해 작업속도도 올라가고 디자인적으로 풍부한 피피티를 만들 수 있습니다. 마지막으로, 양질의 발표를 위해 내용의 근거가 될 수 있는 chart와 table 그리고 image 추천 시스템을 갖추고 있습니다. 내용과 디자인, 모든 면에서 완벽한 피피티를 쉽고 빠르게 만들고 싶은 여러분들에게 텍2피가 발표 프로세스를 획기적으로 바꾸어드릴 것입니다.
'''

# summarized = ps.summarize()[0]
# sentences= ps.split_into_sentences(summarized)
if __name__ == "__main__":
    print("main working")
    ps = processScript(txt1)
    res = ps.main()
    print(res)


