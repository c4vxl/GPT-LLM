# GPT-LLM
This repository contains an implementation of a GPT (Generative Pre-trained Transformer) architecture Language Model (LLM). The model is designed to be flexible and capable of fine-tuning on various datasets using Hugging Face's transformer library.

### Features
- Load pre-trained GPT model from model.pth file.
- Fine-tune the pre-trained model with datasets from Hugging Face.
- Includes a tokenizer that can be loaded from a pre-trained tokenizer available in Hugging Face.
- Basic chatting functionality with the model through the ChatModel class.

### Usage
To use the model, follow these steps:
1. Clone the repository:
2. install dependencies:
   ```
   pip3 install -r requirements.txt
   ```
3. Train & finetune the model with a dataset from huggingface using the `finetune_mdl.py` script.
4. Use the model ```python main.py```


This Program has been created for educational purposes. You can find a documentation on how a Large Language Model works in the `/tutorial/` folder.
- [EN](tutorial/en.md)
- [DE](tutorial/de.md)
