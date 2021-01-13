import numpy as np
import spacy

import torch
import transformers

import config
import dataset
import engine
# from model import BertModel


nlp = spacy.load("en_core_web_sm")

if __name__ == "__main__":
    RUN_PATH = config.BASE_MODEL_PATH+"/finetuned"
    
    sentence = "This place was really great!  I know all Teppanyaki places are very similar, and the \"show\" was typical.  But the food here really was fantastic!  Great fresh veggies and meat!  They also had a sautéed spinach that was part of the meal.  Never seen that before, but it was delicious!  Very reasonable, and large portions.  Their Sake Bombers were also pretty cheap.  Will return for sure!"
#     sentence = "I went out for lunch and I decided to eat a taco at Fridays"
    sentence = " ".join(sentence.split())
    
    word_pieces = [token.text for token in nlp(sentence)]
    tokenized_sentence = config.TOKENIZER.encode(sentence)
    subword_pieces = [config.TOKENIZER.decode(token).replace(" ", "") for token in tokenized_sentence]
    
    print(sentence)
    print(word_pieces)
    print(subword_pieces)
    print(tokenized_sentence)
    
    test_dataset = dataset.EntityDataset(
        texts=[word_pieces],
        tags=[[1] * len(word_pieces)]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = transformers.BertForTokenClassification.from_pretrained(RUN_PATH)
    config = model.config
    model.to(device)
    
    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        loss, logits=model(**data)[:2]
        print(
            [config.id2label[id] for id in logits.argmax(2).cpu().numpy().reshape(-1)[:len(tokenized_sentence)]]
        )
