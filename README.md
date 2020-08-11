A-Catcher
=========
**ADHD 학생들에게 텍스트 자료에 대한 접근 기회의 평등을**
---------------------------------------------------

   <span style="color:blue">ADHD</span> 학생들의 _Attention_을 캐치해서 _A+_을 받도록 도와주는 서비스

      - 내용을 기반으로 하여 파워포인트의 스마트 아트와 도형으로 도식화
      - 발표 대본의 제목과 소제목을 자동으로 추출하여 텍스트를 파워포인트에 삽입
      - 텍스트 주제와 어울리는 피피티 템플릿 및 이미지 추천



### Interactive Web Page for Text2PPTX is coming soon!!


#### Examples
![Alt text](.idea/examplepng.png?raw=true "Title")

#### Dataset collected for BERTsum finetuning
       Presentation Scripts
       약 50건의 발표 대본을 수집하여 문단별로 요약을 위한 라벨링을 함 / Total instances: ~2100 
       
       Naver News Crawling
       정치, 사회, 연예 분야의 뉴스를 크롤링하여 요약을 위한 라벨링을 함
      


#### Dependencies
<pre><code>
   pip install pytorch torchvision -c pytorch (MAC OS)
   pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html (WINDOWS)
   pip install transformers==2.2.2
   pip install neuralcoref
   pip install bert-extractive-summarizer
   pip install python-pptx
   
   </code></pre> 
  
#### Cited
      frameBERT https://github.com/machinereading/frameBERT
      BERT extractive summarization https://github.com/dmmiller612/bert-extractive-summarizer
   

   
   
   
