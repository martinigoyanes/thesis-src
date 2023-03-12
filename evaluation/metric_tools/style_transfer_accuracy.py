from transformers import RobertaTokenizer, RobertaForSequenceClassification
import tqdm
from torch.nn.utils.rnn import pad_sequence


def classify_preds(args, preds, task: str):
    assert task
    print('Calculating style of predictions')
    results = []

    if task == 'jigsaw':
        tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
        model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')
    if task == 'yelp':
        path = '/Midgard/home/martinig/thesis-src/models/roberta_sentiment_classifier'
        tokenizer = RobertaTokenizer.from_pretrained(f"{path}/tokenizer")
        model = RobertaForSequenceClassification.from_pretrained(f"{path}/model")

    for i in tqdm.tqdm(range(0, len(preds), args.batch_size)):
        batch = tokenizer(preds[i:i + args.batch_size], return_tensors='pt', padding=True)
        result = model(**batch)['logits'].argmax(1).float().data.tolist()
        results.extend([1 - item for item in result])

    return results