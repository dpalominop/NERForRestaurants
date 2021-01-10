import transformers


MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = "../models/bert_base_uncased"
MODEL_PATH = "model.bin"
TRAINING_FILE = "../datasets/ner_dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case = True
)
label_types = ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC", "O"]
