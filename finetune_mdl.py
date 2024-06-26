DATASET = "wikitext"
DATASET_CONTEXT = "wikitext-103-raw-v1"

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
import ai
import signal

# exiting logic
def end():
    print("Saving model...")
    model.export_to_file()
    print("Model saved. Exiting.")
    exit(0)

end = lambda: end()

# create tokenizer
tokenizer = ai.Tokenizer()

# define model
modelArgs = ai.ModelArgs().set(vocab_size=tokenizer.vocab_size)
model = ai.Model(modelArgs).to(modelArgs.device)

# load pretrained model from file (if it does exist)
model.load_from_file()

# set hyperparameters
device = torch.device(modelArgs.device)
learning_rate = 3e-5
batch_size = 128
epochs = 5
block_size = modelArgs.block_size

# define optimizer
optimizer = optim.AdamW(model.parameters(), learning_rate)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=block_size, return_tensors='pt')

# load dataset
dataset = load_dataset(DATASET, DATASET_CONTEXT, split='train', trust_remote_code=True)
dataloader = DataLoader(dataset, batch_size=12, shuffle=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# register quit handler
signal.signal(signal.SIGINT, end)

# define collator to handle padding and create language modeling inputs
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Not using masked language modeling
)

# create dataloader
dataloader = DataLoader(tokenized_datasets, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

# training loop
print("Training loop started...")
model.train()
for epoch in range(epochs):
    for batch in dataloader:
        inputs = {key: value.to(device) for key, value in batch.items()}

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs['input_ids'])
        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = inputs['input_ids'][..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # backward pass
        loss.backward()
        
        # optimize
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
    
print("Training finished.")
end