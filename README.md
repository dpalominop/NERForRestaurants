# NER For Restaurants' Reviews
This project use Name Entity Recognition to get insights from Restaurants' reviews.

## Requeriments
- python==3.7.0
- numpy==1.19.2
- pandas==1.2.0
- pytorch==1.7.1
- transformers==4.2.0
- streamlit==0.74.1

(*) For a complete revision, please check *environment.yml* file.

## Virtualenv

### Tested Operative System:

- Windows 10: OK
- All linux-based os: OK

### Steps

1. Clone repository and move to directory:

    ```git clone https://github.com/dpalominop/NERForRestaurants.git && cd NERForRestaurants```
2. Create and activate a virtual environment (I recommend to use conda):

    ```conda create --name ner --file environment.yml```
    
    ```conda activate ner```
3. Run the web application:

    ```streamlit run app.py```
4. Open a browser and write in url:

    ```localhost:8051```

### Only Development Mode

(*) These step are intended only to pretrain o finetune a model from a previous one.

1. Open config.yml and change value of stage to devel:

    ```stage: "devel"```
2. Install git-lfs to run long files:

    ```sudo apt-get install git-lfs```
3. Select a model from https://huggingface.com/models and clone in your local directory:

    ```cd models && git lfs install && git clone https://huggingface.co/{user_name}/{model_name}```
4. Set the pretrained model to use in src/config.py:

    ```BASE_MODEL_PATH = "../models/{model_name}"```
4. Set the dataset to use in src/config.py:

    ```TRAINING_FILE = "../datasets/reviews.csv"```
5. Train your custom model:

    ```cd ../src && python train.py```
6. Use your new custom model to predict tags in a text:

    ```python predict.py```
    
# Demo

Temporarily, the web application will be hosted in https://653a555467d7.ngrok.io

# Help?

Please, contact me to: dapalominop@gmail.com
