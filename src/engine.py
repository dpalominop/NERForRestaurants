import torch
from tqdm import tqdm
from utils import flat_accuracy, annot_confusion_matrix
from dataset import CLS, PAD, SEP

from seqeval.metrics import classification_report


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    tr_loss, tr_accuracy = 0, 0
    tr_preds, tr_labels = [], []
    
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        loss, tr_logits = model(**data)
        loss.backward()
        
        # Subset out unwanted predictions on CLS/PAD/SEP tokens
        preds_mask = (
            (data["ids"] != CLS)
            & (data["ids"] != PAD)
            & (data["ids"] != SEP)
        )
        
#         tr_logits = tr_logits.detach().cpu().numpy()
        tr_label_ids = torch.masked_select(data["labels"], (preds_mask == 1))
        tr_batch_preds = torch.argmax(tr_logits[preds_mask.squeeze()], dim=1)
        tr_batch_labels = tr_label_ids.to("cpu").numpy()
        tr_batch_preds = tr_batch_preds.to("cpu").numpy()
        tr_preds.extend(tr_batch_preds)
        tr_labels.extend(tr_batch_labels)

        # Compute training metrics
        tmp_tr_accuracy = flat_accuracy(tr_batch_labels, tr_batch_preds)
        tr_accuracy += tmp_tr_accuracy

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=1.0
        )
        
        # Update parameters
        optimizer.step()
        scheduler.step()
        tr_loss += loss.item()
        
    tr_loss = tr_loss / len(data_loader)
    tr_accuracy = tr_accuracy / len(data_loader)
    return tr_loss, tr_accuracy

def eval_fn(data_loader, enc_tags, model, device):
    model.eval()
    val_loss, val_accuracy = 0, 0
    val_preds, val_labels = [], []
    
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        loss, val_logits = model(**data)
        
        # Subset out unwanted predictions on CLS/PAD/SEP tokens
        preds_mask = (
            (data["ids"] != CLS)
            & (data["ids"] != PAD)
            & (data["ids"] != SEP)
        )
        
#         val_logits = val_logits.detach().cpu().numpy()
        val_label_ids = torch.masked_select(data["labels"], (preds_mask == 1))
        val_batch_preds = torch.argmax(val_logits[preds_mask.squeeze()], dim=1)
        val_batch_labels = val_label_ids.to("cpu").numpy()
        val_batch_preds = val_batch_preds.to("cpu").numpy()
        val_preds.extend(val_batch_preds)
        val_labels.extend(val_batch_labels)

        # Compute validation metrics
        tmp_val_accuracy = flat_accuracy(val_batch_labels, val_batch_preds)
        val_accuracy += tmp_val_accuracy
        val_loss += loss.item()
        
    # Compute validation reports
    pred_tags = enc_tags.inverse_transform(val_preds)
    val_tags = enc_tags.inverse_transform(val_labels)
    cl_report = classification_report(val_tags, pred_tags)
    conf_mat = None #annot_confusion_matrix(val_tags, pred_tags)

    val_loss = val_loss / len(data_loader)
    val_accuracy = val_accuracy / len(data_loader)
        
    return val_loss, val_accuracy, cl_report, conf_mat
