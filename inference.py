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

# Load the saved model
model = BertForSequenceClassification.from_pretrained("models/model_1000")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print("[*] Model loaded")

# Load and preprocess the test data
df = pd.read_csv("data/review_data.csv")
texts = df["Text"]

# Initialize the BERT tokenizer and tokenize the data
encoded_texts = tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors="pt")

# Create data loader for batching
test_dataset = TensorDataset(
    encoded_texts["input_ids"],
    encoded_texts["attention_mask"]
)
test_loader = DataLoader(test_dataset, batch_size=16)

# Evaluation on the test set
model.eval()
all_embeddings = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Extract the [CLS] token embeddings
        embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()  # Extract the [CLS] token embeddings from the last layer
        all_embeddings.append(embeddings)

all_embeddings = np.concatenate(all_embeddings, axis=0)
print("[*] Embeddings generated")

# Perform PCA
pca = PCA(n_components=3)  # You can adjust the number of components as needed
pca_result = pca.fit_transform(all_embeddings)
print("[*] PCA performed")

# Enable Korean text rendering (seaborn + matplotlib)
font_path = "C:/Windows/Fonts/malgun.ttf"
font_prop = FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()

# EM Clustering (calcuate auto n_components with color map)
n_components = np.arange(1, 21)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(pca_result) for n in n_components]
plt.plot(n_components, [m.bic(pca_result) for m in models], label='BIC')
plt.plot(n_components, [m.aic(pca_result) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
plt.show()

# Use Seaborn for color mapping
colors = sns.color_palette("hls", 3)

# Create a 3D scatter plot using Matplotlib
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=colors, s=10)

# Function to update the view angle for each frame
def update(frame):
    ax.view_init(elev=10, azim=frame * 4)  # Rotate the plot


# Number of frames in the animation
num_frames = 90

# Create the animation
animation = FuncAnimation(fig, update, frames=num_frames, interval=50)

# Save the animation as a GIF file
animation.save("3d_rotation2.gif", writer='imagemagick', fps=10)

# Use Mplcursors to add hover text
cursor = mplcursors.cursor(hover=True)

@cursor.connect("add")
def on_add(sel):
    index = sel.index
    text = df.loc[index, "Text"]
    label = df.loc[index, "Label"]
    # set text with auto newline (if too long)
    sel.annotation.set_text(textwrap.fill(text, 30))


# Show the plot
plt.show()