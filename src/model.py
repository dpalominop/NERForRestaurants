import config
import torch
import transformers
import torch.nn as nn


def loss_func(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss

class EntityModel(nn.Model):
    def __init__(self, num_tags, num_pos):
        super(EntityModel, self).__init__()
        self.num_tags = num_tags
        self.num_pos = num_pos
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH)
        self.bert_drop_1 = nn.Dropout(0.3)
        self.bert_drop_2 = nn.Dropout(0.3)
        self.out_pos = nn.Linear(768, self.num_pos)
        self.out_tag = nn.Linear(768, self.num_tag)
        
    def forward(self, ids, mask, token_type_ids, target_pos, target_tag):
        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        
        bo_pos = self.bert_drop_1(o1)
        bo_tag = self.bert_drop_2(o1)
        
        pos = self.out_pos(bo_pos)
        tag = self.out_tag(bo_tag)
        
        loss_pos = loss_fn(pos, target_pos, mask, self.num_pos)
        loss_tag = loss_fn(tag, target_tag, mask, self.num_tag)
        
        loss = (loss_pos + loss_tag) / 2
        
        return tag, pos, loss
