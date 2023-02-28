# Setup for evaluation
1. ``mkdir -p models/wieting && mkdir -p models/cola && conda activate thesis-src``
2. Wieting similarity model (to evaluate embedding similarities between input and generated text)
```
cd models/wieting
wget https://storage.yandexcloud.net/nlp/wieting_similarity_data.zip
unzip models/wieting/wieting_similarity_data.zip
```
3. RoBERTa fine-tuned for linguistic acceptability on CoLA dataset (to evaluate fluency of generated text)
```
cd models/cola
gdown --folder https://drive.google.com/drive/folders/1p6_3lCbw3J0MhlidvKkRbG73qwmtWuRp
```
4. RoBERTa fine-tuned for toxic classification Jigsaw dataset (to evaluate style transfer)
```
Use model from hugging-face repo: s-nlp/roberta_toxicity_classifier
```