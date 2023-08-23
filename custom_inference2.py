# train with my nn

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from Class.BertWithMCDO import BertWithMCDO


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Load and preprocess the data
df = pd.read_csv("data/naver_shopping_review_data.csv")
texts = df["Text"]

# Initialize the BERT tokenizer and tokenize the data
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
encoded_texts = tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors="pt")


# Initialize the BERT model for sequence classification
model = BertWithMCDO()
model.load('models/bert_monte_carlo_dropout_model(multi-case).pth')

num_samples = 1  # Choose an appropriate number of samples

# add new data frame to save
new_df = pd.DataFrame(columns=["Text","ClusterIdx","PredictedLabel","Mean","Std","EditedStd"])

embeddings = []
cnt = 0

with torch.no_grad():
    for text in texts:
        output_list = []
        for _ in range(num_samples):
            with torch.no_grad():
                inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                outputs, embedding = model(input_ids, attention_mask=attention_mask)
                _, predicted_labels = torch.max(outputs, dim=1)

                output_list.append(outputs)
        
        
        monte_carlo_preds = torch.cat(output_list, dim=0)
        prediction_mean = torch.mean(monte_carlo_preds, dim=0)
        prediction_std = torch.std(monte_carlo_preds, dim=0)
        prediction_entropy = -torch.sum(prediction_mean * torch.log(prediction_mean), dim=0).tolist()

        embeddings.append(embedding)

        prediction_mean = prediction_mean.tolist()

        edited_std = prediction_std - torch.min(prediction_std)
        prediction_label = torch.argmax(torch.mean(monte_carlo_preds, dim=0)).tolist()
        prediction_std = prediction_std.tolist()
        edited_std = edited_std.tolist()

        # add to data frame
        new_df.loc[cnt] = [text, prediction_label, prediction_mean, prediction_std, edited_std]
        cnt += 1

        if (prediction_label == 0):
            print(text)
            print(prediction_label)
            print()


embeddings = np.concatenate(embeddings, axis=0)
# error TypeError: expected Tensor as element 0 in argument 0, but got BaseModelOutputWithPoolingAndCrossAttentions

# Perform PCA on the embeddings
pca = PCA(n_components=3)  # You can adjust the number of components as needed
pca_result = pca.fit_transform(embeddings)

# EMClustering
gmm = GaussianMixture(n_components=3)
gmm.fit(pca_result)
gmm_labels = gmm.predict(pca_result)

# add new columns to data frame
new_df['ClusterIdx'] = gmm_labels

# plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=gmm_labels, s=10)

# save plot
plt.savefig('outputs/embeding_with_custom_test(testdata).png')

# save new data frame
new_df.to_csv('outputs/naver_clustered_results.csv', index=False)
