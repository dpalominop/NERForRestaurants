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
    def __init__(self, model_path, num_labels):
        super(BertModel, self).__init__()
        self.arch = transformers.BertConfig.from_pretrained(
            model_path
        ).architectures[0]
        self.num_labels = num_labels
        
        if self.arch == "BertForMaskedLM":
            self.bert = transformers.BertForTokenClassification.from_pretrained(model_path, num_labels=self.num_labels)

        if self.arch == "BertForTokenClassification":
            self.bert = transformers.BertForTokenClassification.from_pretrained(model_path)
            self.bert.classifier = nn.Linear(self.bert.classifier.in_features, self.num_labels, bias=True)
            self.bert.num_labels = self.num_labels
            
    def set_config(self, enc_tags):
        assert (self.num_labels == len(list(enc_tags.classes_))), "Number of clases in encoder of tags dont match num_labels"
        
        self.bert.config.num_labels = self.num_labels
        self.bert.config.architectures[0] = "BertForTokenClassification"
        self.bert.config.id2label = {str(i):c for i,c in enumerate(enc_tags.classes_)}
        self.bert.config.label2id = {c:str(i) for i,c in enumerate(enc_tags.classes_)}
        
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        loss, logits = [], []
        
        if self.arch == "BertForMaskedLM":
            output = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
            logits = output.logits
            loss = loss_fn(logits, labels, attention_mask, self.num_labels)

        if self.arch == "BertForTokenClassification":
            outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )
            loss, logits = outputs[:2]
        
        return loss, logits
