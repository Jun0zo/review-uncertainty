import pandas as pd
import re
import json

# read data (with column name)
df = pd.read_csv('data/tested_data.csv')
df["OriginalLabel"] = df["OriginalLabel"].apply(lambda x : int(x))
df["PredictedLabel"] = df["PredictedLabel"].apply(lambda x : int(x))
df["SumOfStd"] = df["Std"].apply(lambda x : sum(map(float, json.loads(x))))
df["SumOfMean"] = df["Mean"].apply(lambda x : sum(map(float, json.loads(x))))

original_label = df['OriginalLabel']
predicted_label = df['PredictedLabel']


# Correct VS Wrong

print(" VS ")
# retrieve "EditedStd" that Label and OriginalLabel are same (with original_label and predicted_label)
correct_df = df.loc[original_label == predicted_label]
print('correct std :', sum(correct_df["SumOfStd"]) / len(correct_df))
print('correct mean :', sum(correct_df["SumOfMean"]) / len(correct_df))


# retrieve "EditedStd" that Label and OriginalLabel are different (with original_label and predicted_label)
wrong_df = df.loc[original_label != predicted_label]
print('wrong std :', wrong_df["SumOfStd"].sum() / len(wrong_df))
print('wrong mean :', wrong_df["SumOfMean"].sum() / len(wrong_df))

print("=====================================")

# label 0, 1, 2
df0 = df.loc[original_label == 0]
print('0 std :', df0["SumOfStd"].sum() / len(df0))
print('0 mean :', df0["SumOfMean"].sum() / len(df0))

df1 = df.loc[original_label == 1]
print('1 std :', df1["SumOfStd"].sum() / len(df1))
print('1 mean :', df1["SumOfMean"].sum() / len(df1))

df2 = df.loc[original_label == 2]
print('2 std :', df2["SumOfStd"].sum() / len(df2))
print('2 mean :', df2["SumOfMean"].sum() / len(df2))

print("=====================================")

print(len(df))
print(len(df[(df["OriginalLabel"] == 0) & (df["PredictedLabel"] == 0)]))
print(len(df[(df["OriginalLabel"] == 0) & ((df["PredictedLabel"] == 1) | (df["PredictedLabel"] == 2))]))
print(len(df[((df["OriginalLabel"] == 1) | (df["OriginalLabel"] == 2)) & (df["PredictedLabel"] == 0)]))
print(len(df[((df["OriginalLabel"] == 1) | (df["OriginalLabel"] == 2)) & ((df["PredictedLabel"] == 1) | (df["PredictedLabel"] == 2))]))


print("=====================================")

# sort by sort of "Std" (df["Std"] is list like [1,2,3])
df = df.sort_values(by=['Std'], axis=0, ascending=False)

# save sorted data with only "Text" and "Std"
sorted_df = df[["OriginalLabel", "PredictedLabel", "SumOfStd", "Text"]]
sorted_df.to_csv('results/csv/sorted_data.csv', index=False)

print("=================  Sorted Test  ====================")

# print first row
print("top 10 Text :", df.iloc[:10]["Text"])
print("top 10 Std :", df.iloc[:10]["SumOfStd"])

print("worst 10 Text :",df.iloc[-10:-1]["Text"])
print("worst 10 Std :",df.iloc[-10:-1]["SumOfStd"])