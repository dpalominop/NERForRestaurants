import numpy as np
import spacy

import joblib
import torch

import config
import dataset
import engine
from model import BertModel


nlp = spacy.load("en_core_web_sm")

if __name__ == "__main__":
    RUN_PATH = config.BASE_MODEL_PATH+"/finetuned"
    meta_data = joblib.load(RUN_PATH+"/meta.bin")
    enc_tags = meta_data["enc_tags"]
    
    num_tags = len(list(enc_tags.classes_))
    
    sentence = "This place was really great!  I know all Teppanyaki places are very similar, and the \"show\" was typical.  But the food here really was fantastic!  Great fresh veggies and meat!  They also had a sautéed spinach that was part of the meal.  Never seen that before, but it was delicious!  Very reasonable, and large portions.  Their Sake Bombers were also pretty cheap.  Will return for sure!"
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
    model = BertModel(num_labels=num_tags)
    model.load_state_dict(torch.load(RUN_PATH+"/ch_epoch_9.tar")["model_state_dict"])
    model.to(device)
    
    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        loss, logits=model(**data)
        
        print(
            enc_tags.inverse_transform(
                logits.argmax(2).cpu().numpy().reshape(-1)
            )[:len(tokenized_sentence)]
        )
