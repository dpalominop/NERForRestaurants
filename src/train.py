import pandas as pd
import numpy as np

import joblib
import torch

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
from dataset import TransformerDataLoader
import engine
from model import EntityModel


def process_data(data_path):
    df = pd.read_csv(data_path, encoding="latin-1")
    df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")
    
    enc_tags = preprocessing.LabelEncoder()
    
    df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])
    
    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    tags = df.groupby("Sentence #")["Tag"].apply(list).values
    
    return sentences, tags, enc_tags


if __name__ == "__main__":
    sentences, tags, enc_tags = process_data(config.TRAINING_FILE)
    
    meta_data = {
        "enc_tags": enc_tags
    }
    
    joblib.dump(meta_data, "meta.bin")
    
    num_tags = len(list(enc_tag.classes_))
    
    (
        train_sentences,
        test_sentences,
        train_tags,
        test_tags
    ) = model_selection.train_test_split(sentences, tags, random_state=42, test_size=0.1)
    
    train_data_loader = TransformerDataLoader(texts=train_sentences, tags=train_tags, 
                                              batch_size=config.TRAIN_BATCH_SIZE, num_workers=4)
    valid_data_loader = TransformerDataLoader(texts=test_sentences, tags=test_tags, 
                                              batch_size=config.VALID_BATCH_SIZE, num_workers=1)
    print("Loaded training and validation data into DataLoaders.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EntityModel(num_labels=num_tags)
    model.to(device)
    print(f"Initialized model and moved it to {device}.")
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        }
    ]
    num_train_steps = int(len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    schedule = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    
    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(train_data_loader, model, optimizer, device, schedule)
        test_loss = engine.eval_fn(valid_data_loader, model, device)
        print(f"[{epoch}]: Train Loss = {train_loss} - Valid Loss = {test_loss}")
        if test_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = test_loss
    