
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
import re
import json
#import training data json
with open('training_data.json', 'r') as f:
    training_data = json.load(f)
# Preprocess the training data
texts = [item['text'] for item in training_data]
labels = [item['label'] for item in training_data]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
#model definition
class SimpleClassifer(nn.Module):
    def __init__(self, input_dim):
        super(SimpleClassifer, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.fc2 = nn.Linear(8, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
model = SimpleClassifer(X.shape[1])
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#training loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(torch.tensor(X, dtype=torch.float32))
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()
#prediction function    
def predict_sentiment(input_text):
    # Extract feeling part (robust to different formats)
    feeling_match = re.search(r"(?i)i am .* and (i )?feel(ing)? (.+)", input_text)
    feeling_text = feeling_match.group(3) if feeling_match else input_text

    # Extract name, normalize capitalization 
    name_match = re.search(r"(?i)i am (\w+)", input_text)
    name = name_match.group(1).capitalize() if name_match else "User"

    # Vectorizing only the feeling part
    feeling_vector = vectorizer.transform([feeling_text]).toarray()
    input_tensor = torch.tensor(feeling_vector, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        sentiment = "good" if output.item() > 0.5 else "bad"

    print(f"{name} is feeling {sentiment}.")

#user input
input_text = input("who are you & how are you feeling today?: ")    
predict_sentiment(input_text)
#save the model
torch.save(model.state_dict(), 'sentiment_model.pth')
print("Model saved as sentiment_model.pth")
