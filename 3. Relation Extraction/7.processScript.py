import requests
import re

__all__ = ["Preprocessor", "normalize", "word_tokenize", "sent_tokenize",
           "morph_tokenize", "sent_word_tokenize", "sent_morph_tokenize"]

import functools

_preprocessor = None


class _Mecab(object):
    def __init__(self, dic_path):
        try:
            import MeCab
        except ImportError:
            raise ImportError("could not import `MeCab`; make sure that "
                              "`mecab-python` is installed by running "
                              "`install_mceab_ko.sh` in the repository. ")
        self._dic_path = dic_path
        self._tagger = MeCab.Tagger("-d {}".format(
            dic_path
        ))

    def parse(self, text):
        nodes = self._tagger.parseToNode(text)

        while nodes:
            form = nodes.surface.strip()
            pos = nodes.feature.split(",")[0]
            nodes = nodes.next

            if pos == "BOS/EOS":
                continue

            yield form, pos


class Preprocessor(object):
    def __init__(self):
        self._mecab = None
        self._twitter = None

    def _init_mecab(self):
        self._mecab = _Mecab("/usr/local/lib/mecab/dic/mecab-ko-dic")

        # Run a test to sacrifice two words
        # There is a bug in mecab that makes it omit first two words.
        try:
            _ = list(self._mecab.parse("mecab mecab"))
            del _
        except UnicodeDecodeError:
            pass

    def _init_twitter(self):
        try:
            import twkorean
        except ImportError:
            raise ImportError("could not import `twkorean`; make sure that "
                              "the package is installed by running "
                              "`install_twkorean.sh` in the repository. ")
        self._twitter = twkorean.TwitterKoreanProcessor()

    def normalize(self, text):
        """Normalize a text using open-korean-text.

        Arguments:
            text: text string.

        Returns:
            Normalized text string.
        """
        if self._twitter is None:
            self._init_twitter()

        return self._twitter.normalize(text)

    def word_tokenize(self, text):
        """Tokenize a text into space-separated words.

        This is the most basic form of tokenization, where we do not wish to
        analyze morphology of each individual word.

        Arguments:
            text: text string.

        Returns:
            Generator for a list of space-tokenized words.
        """
        if self._mecab is None:
            self._init_mecab()

        tokens = text.split()
        tokens_it = iter(tokens)

        try:
            token = next(tokens_it)
            index = 0

            for f, pos in self._mecab.parse(text):

                if index >= len(token):
                    if token:
                        yield token

                    token = next(tokens_it)
                    index = 0

                # token_f = token[index:index + len(f)]
                # if token_f != f:
                #     print(token, index, token_f, f)

                if pos.startswith("S"):
                    if index:
                        t, token = token[:index], token[index:]

                        if t:
                            yield t

                        index = 0

                    t, token = token[index:index + len(f)], token[
                                                            index + len(f):]

                    if t:
                        yield t

                    index = 0
                else:
                    index += len(f)

            if token and index:
                yield token[:index]

        except StopIteration:
            pass

    def sent_tokenize(self, text, residual=True):
        """Tokenize a bulk of text into list of sentences (using Mecab-ko).

        Arguments:
            text: text string.
            residual: whether to include an incomplete sentence at the end of
                the text.
        Returns:
            Generator that generates a list of sentence strings in their
            original forms.
        """
        self._init_mecab()

        index = 0

        for f, pos in self._mecab.parse(text):
            index = text.find(f, index)
            index += len(f)

            if pos == "SF":
                sent = text[:index].strip()

                yield sent

                text = text[index:]
                index = 0

        if residual and index > 0:
            sent = text.strip()

            yield sent

    def morph_tokenize(self, text, pos=False):
        """Tokenize a sentence into morpheme tokens (using Mecab-ko).

        Arguments:
            text: sentence string.
            pos: whether to include part-of-speech tags.

        Returns:
            If pos is False, then a generator of morphemes is returned.
            Otherwise, a generator of morpheme and pos tuples is returned.
        """
        self._init_mecab()

        if pos:
            for item in self._mecab.parse(text):
                yield item
        else:
            for f, _ in self._mecab.parse(text):
                yield f

    def sent_morph_tokenize(self, text, residual=True, pos=False):
        """Tokenize a bulk of text into list of sentences (using Mecab-ko).
        Each sentence is a list of morphemes. This is slightly more efficient than
        tokenizing text into sents and morphemes in succession.
        Arguments:
            text: text string.
            residual: whether to include an incomplete sentence at the end of
                the text.
            pos: whether to include part-of-speech tag.
        Returns:
            If pos is False, then a generator of morphemes list is returned.
            Otherwise, a generator of morpheme and pos tuples list is returned.
        """
        self._init_mecab()

        sent = []

        for f, p in self._mecab.parse(text):
            if pos:
                sent.append((f, p))
            else:
                sent.append(f)

            if p == "SF":
                yield sent
                sent = []

        if residual and sent:
            yield sent

    def sent_word_tokenize(self, text, residual=True):
        """Tokenize a bulk of text into list of sentences (using Mecab-ko).
        Each sentence is a list of words. This is slightly more efficient than
        tokenizing text into sents and words in succession.
        Arguments:
            text: text string.
            residual: whether to include an incomplete sentence at the end of
                the text.
        Returns:
            A generator of words list.
        """
        self._init_mecab()

        sent = []
        tokens = text.split()
        tokens_it = iter(tokens)

        try:
            token = next(tokens_it)
            index = 0
            yield_sent = False

            for f, pos in self._mecab.parse(text):

                if index >= len(token):
                    if token:
                        sent.append(token)

                    token = next(tokens_it)
                    index = 0

                    if yield_sent:
                        yield sent
                        yield_sent = False
                        sent = []

                # assert token[index:index + len(f)] == f

                if pos.startswith("S"):
                    if index:
                        t, token = token[:index], token[index:]

                        if t:
                            sent.append(t)

                        index = 0

                    t, token = token[index:index + len(f)], token[
                                                            index + len(f):]

                    if t:
                        sent.append(t)

                    index = 0

                    if pos == "SF":
                        yield_sent = True
                else:
                    index += len(f)

            if token and index:
                sent.append(token[:index])

            if sent and (yield_sent or residual):
                yield sent

        except StopIteration:
            pass


def normalize(text, *args, **kwargs):
    global _preprocessor

    if _preprocessor is None:
        _preprocessor = Preprocessor()

    return _preprocessor.normalize(text, *args, **kwargs)


def morph_tokenize(text, *args, **kwargs):
    global _preprocessor

    if _preprocessor is None:
        _preprocessor = Preprocessor()

    return _preprocessor.morph_tokenize(text, *args, **kwargs)


def sent_tokenize(text, *args, **kwargs):
    global _preprocessor

    if _preprocessor is None:
        _preprocessor = Preprocessor()

    return _preprocessor.sent_tokenize(text, *args, **kwargs)


def word_tokenize(text, *args, **kwargs):
    global _preprocessor

    if _preprocessor is None:
        _preprocessor = Preprocessor()

    return _preprocessor.word_tokenize(text, *args, **kwargs)


def sent_word_tokenize(text, *args, **kwargs):
    global _preprocessor

    if _preprocessor is None:
        _preprocessor = Preprocessor()

    return _preprocessor.sent_word_tokenize(text, *args, **kwargs)


def sent_morph_tokenize(text, *args, **kwargs):
    global _preprocessor

    if _preprocessor is None:
        _preprocessor = Preprocessor()

    return _preprocessor.sent_morph_tokenize(text, *args, **kwargs)


functools.update_wrapper(normalize, Preprocessor.normalize)
functools.update_wrapper(sent_tokenize, Preprocessor.sent_tokenize)
functools.update_wrapper(morph_tokenize, Preprocessor.morph_tokenize)
functools.update_wrapper(word_tokenize, Preprocessor.word_tokenize)
functools.update_wrapper(sent_word_tokenize, Preprocessor.sent_word_tokenize)
functools.update_wrapper(sent_morph_tokenize, Preprocessor.sent_morph_tokenize)


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


