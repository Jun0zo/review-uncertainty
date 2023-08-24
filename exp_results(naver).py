import pandas as pd
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def z_score_normalize(data):
    mean = sum(data) / len(data)
    std = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
    normalized_data = [(x - mean) / std for x in data]
    return normalized_data

# read data (with column name)
df = pd.read_csv('results/mid/tested_data(naver-light).csv')
df["PredictedLabel"] = df["PredictedLabel"].apply(lambda x : int(x))

df["Std"] = df["Std"].apply(lambda x : list(map(float, json.loads(x))))
df["Mean"] = df["Mean"].apply(lambda x : list(map(float, json.loads(x))))

# df["Mean"] = df["Mean"].apply(z_score_normalize)
# df["Std"] = df["Std"].apply(z_score_normalize)

# get new Std from std that x - min(x)
df["EditedStd"] = df["Std"].apply(lambda x : [i - min(x) for i in x])
df["SumOfEditedStd"] = df["EditedStd"].apply(lambda x : sum(x))

df["SumOfStd"] = df["Std"].apply(lambda x : sum(x))
df["SumOfMean"] = df["Mean"].apply(lambda x : sum(x))

# get new Std from Std that predicted label
df["NewStd"] = df.apply(lambda x : x["Std"][x["PredictedLabel"]], axis=1)
df["NewMean"] = df.apply(lambda x : x["Mean"][x["PredictedLabel"]], axis=1)

# get new Std from Std that except predicted label
df["ExceptStd"] = df.apply(lambda x : x["Std"][:x["PredictedLabel"]] + x["Std"][x["PredictedLabel"]+1:], axis=1)
df["ExceptMean"] = df.apply(lambda x : x["Mean"][:x["PredictedLabel"]] + x["Mean"][x["PredictedLabel"]+1:], axis=1)



# z-score
df["Z-Score"] = df["Std"].apply(lambda x : (x - np.mean(x)) / np.std(x))
df["SumOfZ-Score"] = df["Z-Score"].apply(lambda x : sum(x))

predicted_label = df['PredictedLabel']


print("=================  Sorted Test  ====================")

# sort by SumOfStd
df = df.sort_values(by=['SumOfEditedStd'], axis=0, ascending=False)

# print first row
print("top 10 Text :", df.iloc[:10]["Text"])
print("top 10 Std :", df.iloc[:10]["NewStd"])

print("worst 10 Text :",df.iloc[-10:-1]["Text"])
print("worst 10 Std :",df.iloc[-10:-1]["NewStd"])

df.to_csv('outputs/tested_data(naver-sorted).csv', index=False)

print("=================  Sorted Test  ====================")

plt.figure(figsize=(10, 10))
sns.set(font_scale=1.5)

# i want to plot data with "0" to high, "1" to middle, "2" to low (scatter plot)
# so i set "hue" to "OriginalLabel"
sns.scatterplot(x=df.index, y=df["SumOfEditedStd"], hue=df["PredictedLabel"], palette="Set2")

plt.xlabel('Index')
plt.ylabel('Sum of Std')
plt.title('Sorted Std')
plt.legend()
plt.grid(True)
plt.savefig('outputs/sorted_test(naver-light).png')