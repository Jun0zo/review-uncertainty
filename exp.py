import pandas as pd
import re
import json
import numpy as np

read_csv_path = 'results/csv/sorted_data2(uncasead-base).csv'

df = pd.read_csv(read_csv_path)
df.sort_values(by=['SumOfStd'], axis=0, inplace=True, ascending=False)

# slice top 200
df = df[-201:]

# plot Mean, Std, middle, q1, q3, iqr, upper, lower of "SumOfStd" each label
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=2.5)
plt.figure(figsize=(10, 10))

# label 0, 1, 2
df0 = df.loc[df["OriginalLabel"] == 0]
df1 = df.loc[df["OriginalLabel"] == 1]
df2 = df.loc[df["OriginalLabel"] == 2]

# plot
plt.boxplot([df0["SumOfStd"], df1["SumOfStd"], df2["SumOfStd"]], labels=["High", "Middle", "Low"])

# make as table
df0 = df0["SumOfStd"].describe()
df1 = df1["SumOfStd"].describe()
df2 = df2["SumOfStd"].describe()
df0 = df0.to_frame().T
df1 = df1.to_frame().T
df2 = df2.to_frame().T
df0["Label"] = "Low"
df1["Label"] = "Middle"
df2["Label"] = "High"
new_df = pd.concat([df0, df1, df2])
new_df = new_df[["Label", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
new_df.to_csv('outputs/SumOfStd.csv', index=False)
print(new_df)


plt.xlabel('Label')
plt.ylabel('Std')
# plt.title('SumOfStd')
plt.legend()
plt.grid(True)

plt.savefig('outputs/SumOfStd.png')

# count each label
print('len : ', len(df))
label_counts = df['OriginalLabel'].value_counts()
print(label_counts)



# plot acc, precision, recall, f1 at bar (with 4 difference colors 'red', 'blue', 'green', 'yellow' softly) (with tag text)
# import matplotlib.pyplot as plt
# import numpy as np  
# import seaborn as sns

# plt.figure(figsize=(10, 10))
# sns.set(font_scale=2.5)

# x = np.arange(4)
# width = 0.5


# acc = 95 / 100
# precision = 394 / (394 + 5)
# recall = 394 / (394 + 10)
# f1 = 2 * (precision * recall) / (precision + recall)

# # plot acc, precision, recall, f1 at bar (use colormap_
# plt.bar(x, [acc, precision, recall, f1], width, color=sns.color_palette("Set1"))

# # add value text
# for i, v in enumerate([acc, precision, recall, f1]):
#     plt.text(i - 0.1, v + 0.01, str(round(v, 3)), color='black', fontweight='bold')

# plt.xticks(x, ('Accuracy', 'Precision', 'Recall', 'F1'))
# plt.xlabel('Metrics')
# plt.ylabel('Value')

# plt.savefig('outputs/acc_precision_recall_f1.png')
