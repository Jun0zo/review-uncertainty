import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mplcursors
import textwrap
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
from matplotlib.font_manager import FontProperties
from matplotlib.animation import FuncAnimation
from PIL import Image
from sklearn.mixture import GaussianMixture

dropout_rate = 0.3

# Load the saved model
model = BertForSequenceClassification.from_pretrained("models/model_1000", classifier_dropout=0.5)
model.cuda()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print("[*] Model loaded")

model.eval()

num_samples = 5  # Choose an appropriate number of samples
predictions = []

text = "컬러 이쁘네요~ 블랙손잡이바디가 갖고싶어서 여러개보던중에 맘에들어서 구매했어요~ 내구성은 좀 써봐야 확실히알겠지만 캐리어만봐도 여행전 설레이는기분이 더 드네요 ^^ 가까이찍은사진있으니 자세히 보세요~"

for _ in range(num_samples):
    with torch.no_grad():
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs, output_attentions=False)
        logits = outputs.logits
        _, predicted_labels = torch.max(logits, dim=1)
        print(predicted_labels)
        print(logits)
