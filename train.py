import json 
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open('intents.json','r') as f:
    intents=json.load(f)
    
all_words=[]
tags=[]
xy=[]

for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))
              
ignore_words=["?","!",".",","]
all_words=[stem(w) for w in all_words if w not in ignore_words]
tags=sorted(set(tags))
all_words=sorted(set(all_words))

x_train=[]
y_train=[]

for (pattern_sentence,tag) in xy:
     bag=bag_of_words(pattern_sentence, all_words )
     x_train.append(bag)
     
     label=tags.index(tag)
     y_train.append(label )
     
     
x_train=np.array(x_train)
y_train=np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
batch_size=8
hidden_size=8
output_size=len(tags)
input_size=len(x_train[0])
learning_rate=0.0001
num_epochs=1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        loss.requires_grad = True
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')  

data={
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}
 
FILE = "data.pth"
torch.save(data,  FILE)
 
print(f'training complete. File saved to {FILE}')