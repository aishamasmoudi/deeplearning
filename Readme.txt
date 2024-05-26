An Investigation of Large Language Models for Multilingual Hate Speech Detection:

This folder contain the following file:

    Bert folder
        - preprocess.ipynb
        - train_model.ipynb
        - train_model.py
    
    dataset folder
        - ar_dataset_600_translated.csv
        - ar_dataset_600.csv
        - ar_dataset.csv 
        - en_dataset_600.csv 
        - en_dataset.csv 
        - fr_dataset_600_translated.csv 
        - fr_dataset_600.csv 
        - fr_dataset.csv

    detxify folder
        - detoxify.ipynb

    - run_LLM.ipynb
    - translation.ipynb


How to run LLM:
All infos can be found here: https://python.langchain.com/docs/integrations/llms/ollama/
    - download ollama (more info: https://ollama.com/library)
    - In command prompt type 'ollama pull <model name>'
    - run the file 'run_LLM.ipynb'
    - Make sure to dowload all the library and to chose the right model and prompt.


About our data:
    The original dataset can be found here: https://huggingface.co/datasets/nedjmaou/MLMA_hate_speech
    We preprocess this dataset to get our CSV files in the dataset folder.
    Note that initially we had 600 tweets with 3 categories: normal, offensive and hateful. 
    We finally chose to only have 2 categories: normal and hateful. So we only use 400 out of the 600 tweets in our datasets.
    The conversion from 600 to 400 tweet is done inside our codes.



