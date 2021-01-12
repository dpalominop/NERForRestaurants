import transformers
import datetime

MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = "../models/bert-large-NER"
TRAINING_FILE = "../datasets/reviews.csv"
BERT_ARCH = transformers.BertConfig.from_pretrained(
    BASE_MODEL_PATH
).architectures[0]
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case = True
)
label_types = ["B-RES", "I-RES", "B-DIS", "I-DIS", "B-OCC", "I-OCC", "O"]
THIS_RUN = datetime.datetime.now().strftime("%Y%m%d-%H.%M.%S")
