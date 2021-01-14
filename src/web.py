# -*- coding: UTF-8 -*-
from bokeh.models.widgets.markups import Div
import boto3
import numpy as np
import pandas as pd
import os
from transformers import BertTokenizer, BertForTokenClassification, BertConfig
import streamlit as st
import torch


def get_bert_pred_df(model, tokenizer, input_text, label_types):

    """
    Uses the model to make a prediction, with batch size 1.
    """

    encoded_text = tokenizer.encode(input_text)
    wordpieces = [tokenizer.decode(tok).replace(" ", "") for tok in encoded_text]

    inputs = tokenizer(input_text, return_tensors="pt")
    labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1
    outputs = model(**inputs, labels=labels)
    loss, scores = outputs[:2]
    scores = scores.detach().numpy()

    label_ids = np.argmax(scores, axis=2)
    preds = [model.config.id2label[id] for id in label_ids[0]]

    wp_preds = list(zip(wordpieces[1:-1], preds[1:-1]))
    toplevel_preds = [pair[1] for pair in wp_preds if "##" not in pair[0]]
    str_rep = " ".join([t[0] for t in wp_preds]).replace(" ##", "").split()

    # If resulting string length is correct, create prediction columns for each tag
    if len(str_rep) == len(toplevel_preds):
        preds_final = list(zip(str_rep, toplevel_preds))
        b_preds_df = pd.DataFrame(preds_final)
        b_preds_df.columns = ["text", "pred"]
        for tag in label_types:
            b_preds_df[f"{tag}"] = np.where(
                b_preds_df["pred"].str.contains(tag), 1, 0
            )
        return b_preds_df.loc[:, "text":]
    else:
        print("Could not match up output string with preds.")
        return None

@st.cache()
def load_model_and_tokenizer(chk_path, state_dict_name="model_state_dict"):

    """
    Loads model from a specified checkpoint path. Replace `CHK_PATH` at the top of
    the script with where you are keeping the saved checkpoint.
    (Checkpoint must include a model state_dict, which by default is specified as
    'model_state_dict,' as it is in the `main.py` script.)
    """

    model_path = "/".join(chk_path.split("/")[:-1])
    token_path = "/".join(chk_path.split("/")[:-2])
    
    model = BertForTokenClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(token_path, do_lower_case=False)

    return model, tokenizer

class LanguageResourceManager:

    """
    Manages resources for each language, such as the models. Also acts as a
    convenient interface for getting predictions.
    """

    def __init__(self, config, chk_path):
        self.label_types = config["label_types"]
        self.num_labels = len(self.label_types)
        
        self.bert_model, self.bert_tokenizer = load_model_and_tokenizer(
            chk_path
        )

    def get_preds(self, input_text):
        return get_bert_pred_df(
            self.bert_model, self.bert_tokenizer, input_text, self.label_types
        )

def create_explainer(color_dict, ent_dict):

    explainer = """<b>Note:</b> Each Tag is associated wiht one different color.<br><br>"""

    for ent_type in ent_dict:
        dark, light = color_dict[ent_dict[ent_type]]
        ent_html = f"""<b><span style="color: {dark}">{ent_type}</span></b><br>"""
        explainer += ent_html

    return Div(text=explainer, width=500)

def produce_text_display(pred_df, color_dict, label_types):

    """
    Returns a bokeh Div object containing the prediction DF's text as formatted
    HTML. The color of the word corresponds to the entity type, as defined in
    `color_dict` (which right now is pulled from `demo_colors` in config.yaml)
    Right now, `tooltip` is set to False by default because it is not supported
    in Streamlit.
    """

    def style_wrapper(s, tag, pred_score, tooltip=False):
        # Wraps a word that at least one model predicted to be an entity.
        dark, light = color_dict[tag]
        color = dark

        # note to self: using the "dark" color for the bg-color of the text is
        # generally too dark. Change this once hovers/tooltips are supported in
        # Streamlit.
        if tooltip:
            long_tag_names = {  # Define longer tag names for tooltip clarity
                "res": "RESTAURANT",
                "dis": "DISH",
                "occ": "OCCASION",
            }
            html = f"""<span class="pred" style="background-color: {color}">
            <span class="tooltip">
            {s}
            <span class="tooltiptext" style="background-color: {color}">
                <b>{long_tag_names[tag]}</b>
            </span>
            </span>
            </span>"""
        else:  # Simply change the inline color of the predicted word
            html = f"""<span style="color: {color}; font-weight: bold">
            {s}</span>
            """

        return html.replace("\n", "")

    text = []
    ps_cols = [col for col in pred_df.columns if col in label_types]
    
    # Iterates over each piece of text in the preds df, checking whether the text
    # was predicted to be at least one entity type. Grabs the tag name off the
    # end of the column name, then passes the necessary args to `style_wrapper`
    # to wrap that particular word in the styling HTML.
    for i, row in pred_df.iterrows():

        if row[ps_cols].sum() > 0:
            row_no_text = row[ps_cols]
            tag_col = row_no_text[row_no_text > 0].index[0]
            tag = tag_col.split("-")[1].lower()
            wrapped_text = style_wrapper(
                row["text"], tag, row[f"{tag_col}"]
            )
            text.append(wrapped_text)
        else:
            text.append(row["text"])

    # Future: once tooltips are supported, add tooltip CSS here.
    html_string = (
        """<div style="font-size: 18px; border-color: black">"""
        + " ".join(text)
        + "</div>"
    )

    return Div(text=html_string, width=700)
