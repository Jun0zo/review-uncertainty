import pandas as pd
import re
import json
import numpy as np
# read data (with column name)
# df = pd.read_csv('data/tested_data(base).csv')

read_csv_path = 'results/mid/tested_data(song-light-last-max).csv'
sorted_csv_path = 'results/csv/sorted_data(light-last-max).csv'


df = pd.read_csv(read_csv_path)
df["OriginalLabel"] = df["OriginalLabel"].apply(lambda x : int(x))
df["PredictedLabel"] = df["PredictedLabel"].apply(lambda x : int(x))

original_label = df['OriginalLabel']
predicted_label = df['PredictedLabel']

# Correct VS Wrong

print(" VS ")
# retrieve "EditedStd" that Label and OriginalLabel are same (with original_label and predicted_label)
correct_df = df.loc[original_label == predicted_label]
print('correct std :', sum(correct_df["Std"]) / len(correct_df))
print('correct mean :', sum(correct_df["Mean"]) / len(correct_df))


# retrieve "EditedStd" that Label and OriginalLabel are different (with original_label and predicted_label)
wrong_df = df.loc[original_label != predicted_label]
print('wrong std :', wrong_df["Std"].sum() / len(wrong_df))
print('wrong mean :', wrong_df["Mean"].sum() / len(wrong_df))

print("=====================================")

# label 0, 1, 2
df0 = df.loc[original_label == 0]
print('0 std :', df0["Std"].sum() / len(df0))
print('0 mean :', df0["Mean"].sum() / len(df0))
# print('0 z-score :', df0["SumOfZ-Score"].sum() / len(df0))

df1 = df.loc[original_label == 1]
print('1 std :', df1["Std"].sum() / len(df1))
print('1 mean :', df1["Mean"].sum() / len(df1))
# print('1 z-score :', df1["SumOfZ-Score"].sum() / len(df1))

df2 = df.loc[original_label == 2]
print('2 std :', df2["Std"].sum() / len(df2))
print('2 mean :', df2["Mean"].sum() / len(df2))
# print('2 z-score :', df2["SumOfZ-Score"].sum() / len(df2))



print("=====================================")

print(len(df))
print(len(df[(df["OriginalLabel"] == 0) & (df["PredictedLabel"] == 0)]))
print(len(df[(df["OriginalLabel"] == 0) & ((df["PredictedLabel"] == 1) | (df["PredictedLabel"] == 2))]))
print(len(df[((df["OriginalLabel"] == 1) | (df["OriginalLabel"] == 2)) & (df["PredictedLabel"] == 0)]))
print(len(df[((df["OriginalLabel"] == 1) | (df["OriginalLabel"] == 2)) & ((df["PredictedLabel"] == 1) | (df["PredictedLabel"] == 2))]))

print('4 metrix te', df[(df["OriginalLabel"] == 2) & (df["PredictedLabel"] == 1)][:5][["OriginalLabel", "PredictedLabel", "Text"]])
for t in df[(df["OriginalLabel"] == 2) & (df["PredictedLabel"] == 1)][:5][["Text"]].values.tolist():
    print(t)

print("=====================================")

# sort by sort of "Std" (df["Std"] is list like [1,2,3])
df = df.sort_values(by=['Std'], axis=0, ascending=False)

print(df.head(10))
# save sorted data with only "Text" and "Std"
sorted_df = df[["OriginalLabel", "PredictedLabel", "Std", "Text"]]
sorted_df.to_csv(sorted_csv_path, index=False)

print("=================  Sorted Test  ====================")

# plot sorted data
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 10))
sns.set(font_scale=1.5)

# i want to plot data with "0" to high, "1" to middle, "2" to low (scatter plot)
# so i set "hue" to "OriginalLabel"
sns.scatterplot(x=sorted_df.index, y=sorted_df["Std"], hue=sorted_df["OriginalLabel"], palette="Set2")

plt.xlabel('Index')
plt.ylabel('Sum of Std')
plt.title('Sorted Std')
plt.legend()
plt.grid(True)
plt.savefig('outputs/sorted_test(multi).png')


print("=================  Sorted Test  ====================")


# print first row
print("top 10 Text :", df.iloc[:10]["Text"])
print("top 10 Std :", df.iloc[:10]["Std"])

print("worst 10 Text :",df.iloc[-10:-1]["Text"])
print("worst 10 Std :",df.iloc[-10:-1]["Std"])

print("=====================================================")

# get confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(original_label, predicted_label)
print(cm)

# plot confusion matrix (font size big)
plt.figure(figsize=(10, 10))
sns.set(font_scale=2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('outputs/confusion_matrix.png')

# get classification report
from sklearn.metrics import classification_report
print(classification_report(original_label, predicted_label))

# get accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(original_label, predicted_label))

print("=====================================================")
from sklearn.metrics import precision_recall_curve

precision = dict()
recall = dict()
thresholds = dict()
for i in range(3):  # Assuming you have 3 classes (0, 1, 2)
    precision[i], recall[i], thresholds[i] = precision_recall_curve(
        (original_label == i).astype(int),
        (predicted_label == i).astype(int),
    )

# Plot precision-recall curves for class only 0 and 2
plt.figure(figsize=(10, 10))
sns.set(font_scale=1.5)
plt.plot(recall[0], precision[0], lw=2, label='Precision-Recall curve for class 0')
plt.plot(recall[2], precision[2], lw=2, label='Precision-Recall curve for class 2')
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.savefig('outputs/precision_recall_curve.png')

print("=====================================================")

# plot sorted data
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 10))
sns.set(font_scale=1.5)

# i want to plot data with "0" to high, "1" to middle, "2" to low (scatter plot)
# so i set "hue" to "OriginalLabel"
sns.scatterplot(x=sorted_df.index, y=sorted_df["Std"], hue=sorted_df["OriginalLabel"], palette="Set2")

plt.xlabel('Index')
plt.ylabel('Sum of Std')
plt.title('Sorted Std')
plt.legend()
plt.grid(True)
plt.savefig('outputs/sorted_test(multi).png')

acc = 0.957
precision = 394 / (394 + 5)
recall = 394 / (394 + 11)
f1 = 2 * (precision * recall) / (precision + recall)

# plot acc, precision, recall, f1 at bar (with 4 difference soft colors, not very bright) (with tag text)
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 10))
sns.set(font_scale=1.5)

x = np.arange(4)
width = 0.5

plt.bar(x, [acc, precision, recall, f1], width, color=['#e6e6e6', '#cccccc', '#b3b3b3', '#999999'])
plt.xticks(x, ('Accuracy', 'Precision', 'Recall', 'F1'))
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Metrics')


plt.savefig('outputs/acc_precision_recall_f1.png')

print("=====================================================")


# get normal distribution of "Std" each label
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 10))
sns.set(font_scale=1.5)

sns.distplot(df0["Std"], hist=False, label="Low")
sns.distplot(df1["Std"], hist=False, label="Middle")
sns.distplot(df2["Std"], hist=False, label="High")

plt.xlabel('Std')
plt.ylabel('Density')
plt.title('Std Distribution')
plt.legend()
plt.savefig('outputs/Stds/std_distribution.png')

