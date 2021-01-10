import config
import torch

CLS = config.TOKENIZER.vocab["[CLS]"]
PAD = config.TOKENIZER.vocab["[PAD]"]
SEP = config.TOKENIZER.vocab["[SEP]"]


class EntityDataset:
    def __init__(self, texts, tags):
        self.texts = texts
        self.tags = tags
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        itext = self.texts[item]
        itags = self.tags[item]
        
        ids = []
        target_tags = []
        
        for i,w in enumerate(itext):
            inputs = config.TOKENIZER.encode(
                w,
                add_special_tokens = False
            )
            
            input_len = len(inputs)
            ids.extend(inputs)
            target_tags.extend([itags[i]] * input_len)
            
        ids = ids[:config.MAX_LEN - 2]
        target_tags = target_tags[:config.MAX_LEN - 2]
        
        ids = [CLS] + ids + [SEP]
        target_tags = [0] + target_tags + [0]
        
        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)
        
        padding_len = config.MAX_LEN - len(ids)
        
        ids = ids + ([PAD] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)
        
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long)
        }

class TransformerDataloader(torch.utils.data.DataLoader):
    def __init__(self, texts, tags, batch_size=32, num_workers=1):
        entity_dataset = EntityDataset(texts, tags)
        super(TransformerDataset, self).__init__(dataset=entity_dataset, batch_size=batch_size, num_workers=num_workers)
        
        