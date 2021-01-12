import config
import torch
import transformers
import torch.nn as nn


def loss_fn(output, target, mask, num_labels):
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


class BertModel(nn.Module):
    def __init__(self, num_labels):
        super(BertModel, self).__init__()
        
        self.num_labels = num_labels
        
        if config.BERT_ARCH == "BertForMaskedLM":
            self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH)
            self.dropout = nn.Dropout(0.3)
            self.classifier = nn.Linear(self.bert.bert.pooler.dense.out_features, self.num_labels)

        if config.BERT_ARCH == "BertForTokenClassification":
            self.bert = transformers.BertForTokenClassification.from_pretrained(config.BASE_MODEL_PATH, num_labels=9)
            self.bert.classifier = nn.Linear(self.bert.classifier.in_features, self.num_labels, bias=True)
            self.bert.num_labels = self.num_labels
        
    def forward(self, ids, mask, token_type_ids, labels):
        loss, logits = [], []
        
        if config.BERT_ARCH == "BertForMaskedLM":
            output, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)        
            output_dropped = self.dropout(output)
            logits = self.classifier(output_dropped)
            loss = loss_fn(logits, labels, mask, self.num_labels)

        if config.BERT_ARCH == "BertForTokenClassification":
            outputs = self.bert(
                    ids,
                    token_type_ids=None,
                    attention_mask=mask,
                    labels=labels,
                )
            loss, logits = outputs[:2]
        
        return loss, logits
