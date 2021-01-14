# -*- coding: UTF-8 -*-
import argparse
import os
import streamlit as st
from src.web import (
    LanguageResourceManager,
    create_explainer,
    produce_text_display,
)
import yaml

# Run using `streamlit run demo.py en`

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg",
        "--config",
        dest="config_dir",
        type=str,
        default="config.yml",
        help="where the config file is located",
    )
    args = parser.parse_args()

#     Set up configuration (see config.yml)
    with open(args.config_dir, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # App functionality code begins here
    st.title("NER For Restauntants' Reviews")

    # Switch between devel and deploy models
    available_chkpts = []
    if cfg["stage"] == "deploy":
        hf_model = cfg["model_to_deploy"]
        hf_model_path = f"models/"+hf_model.split("/")[1]
        if not os.path.exists(hf_model_path):
            os.system(f"cd models && git lfs install && git clone https://huggingface.co/{hf_model}")
        available_chkpts.extend([hf_model_path])
    else:
        available_chkpts.extend([
            f"models/{dir}/finetuned/" + chkpt
            for dir in os.listdir("models/")
            if dir not in [".keep", ".ipynb_checkpoints"] and os.path.exists(f"models/{dir}/finetuned/") and os.listdir(f"models/{dir}/finetuned/")
            for chkpt in os.listdir(f"models/{dir}/finetuned/") if chkpt not in ["config.json", ".ipynb_checkpoints"]
        ])

    # Create selectbox for users to select checkpoint
    CHK_PATH = st.selectbox("Model checkpoint:", tuple(available_chkpts))

    try:
        mgr = LanguageResourceManager(cfg, CHK_PATH)
    except RuntimeError:
        st.write("The selected checkpoint is not compatible with this BERT model.")
        st.write("Are you sure you have the right checkpoint?")

    user_prompt = "What text do you want to predict on?"
    default_input = cfg["default_text"]
    user_input = st.text_area(user_prompt, value=default_input)

    # Produce and align predictions
    bert_preds = mgr.get_preds(user_input)

    st.subheader("Prediction Summary:")

    # Set up colors and HTML for the explainer and the predicted text
    color_dict = cfg["demo_colors"]
    ent_dict = {
        "Restaurant": "res",
        "Dish": "dis",
        "Occasion": "occ",
    }
    display = produce_text_display(bert_preds, color_dict, mgr.label_types)
    explainer = create_explainer(color_dict, ent_dict)
    ent_types = list(ent_dict.keys())

    # Display the explainer and predicted text
    st.bokeh_chart(explainer)
    st.bokeh_chart(display)

    st.subheader("Prediction Details Per Entity Type:")
    
    # Display fine-grained model prediction columns for selected entity
    mask = (bert_preds[f"B-RES"].values > 0) | (bert_preds[f"I-RES"].values > 0) | \
            (bert_preds[f"B-DIS"].values > 0) | (bert_preds[f"I-DIS"].values > 0) | \
            (bert_preds[f"B-OCC"].values > 0) | (bert_preds[f"I-OCC"].values > 0)
    st.table(bert_preds[mask][["text", f"B-RES", f"I-RES", f"B-DIS", f"I-DIS", f"B-OCC", f"I-OCC"]])
