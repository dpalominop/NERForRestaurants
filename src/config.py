import transformers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 12
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = "../models/bert-large-NER"
TRAINING_FILE = "../datasets/reviews.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case = True
)
THIS_RUN = "finetuned"
