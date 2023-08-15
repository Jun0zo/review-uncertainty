import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer

class BertForSequenceClassificationWithMonteCarloDropout(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(BertForSequenceClassificationWithMonteCarloDropout, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, input_ids, attention_mask):
        # Apply dropout to the input embeddings
        logits = self.bert(input_ids, attention_mask=attention_mask)[0]
        print(logits)
        logits_with_dropout = self.dropout(logits)
        print(logits_with_dropout)
        return logits_with_dropout

# Initialize the model
model = BertForSequenceClassificationWithMonteCarloDropout(dropout_rate=0.7)

# Tokenize input
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
input_text = "컬러 이쁘네요~ 블랙손잡이바디가 갖고싶어서 여러개보던중에 맘에들어서 구매했어요~ 내구성은 좀 써봐야 확실히알겠지만 캐리어만봐도 여행전 설레이는기분이 더 드네요 ^^ 가까이찍은사진있으니 자세히 보세요~"

input_ids = tokenizer.encode(input_text, add_special_tokens=True)
input_ids = torch.tensor(input_ids).unsqueeze(0)  # Batch dimension added
attention_mask = (input_ids != 0).float()

# Perform Monte Carlo Inference
num_mc_samples = 100
logits_list = []

model.eval()

for i in range(num_mc_samples):
    with torch.no_grad():
        model.eval()
        torch.manual_seed(i)
        logits = model(input_ids, attention_mask)
        logits_list.append(logits)

# Calculate mean and standard deviation of logits
logits_list = torch.cat(logits_list, dim=0)
mean_logits = torch.mean(logits_list, dim=0)
std_logits = torch.std(logits_list, dim=0)

print(logits_list)
print(mean_logits)
print(std_logits)
