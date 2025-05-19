import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import os
import zipfile

# Load the data
data = {
    "Query": [
        "How do I reset my password?",
        "My device won't turn on. What should I do?",
        "What is the warranty period for your products?",
        "Can I update the firmware manually?",
        "How to connect my device to Wi-Fi?",
        "What is your return policy?",
        "I forgot my account username, how do I retrieve it?",
        "Is this product compatible with Windows 11?",
        "What are the working hours for customer support?",
        "Tell me a joke",
        "Who won the World Cup in 2022?",
        "How to resset my pasword?",
        "Steps for password reset"
    ],
    "Response Time (s)": [
        2.93, 0.61, 0.49, 0.74, 0.97, 0.60, 0.74, 0.47, 0.97, 0.94, 0.53, 1.21, 0.74
    ],
    "Relevance Score": [
        0.67, 0.33, 0.0, 0.25, 0.75, 0.0, 0.0, 0.0, 0.0, None, None, 0.0, 1.0
    ],
    "Cosine Similarity": [
        0.78, 0.43, 0.22, 0.23, 0.68, 0.18, 0.65, 0.24, 0.17, 0.11, 0.14, 0.33, 0.74
    ],
    "Response Length": [
        10, 4, 2, 6, 11, 3, 8, 1, 5, 5, 1, 13, 8
    ]
}

df = pd.DataFrame(data)

# Label queries as Query A, B, C, ...
labels = list(string.ascii_uppercase)[:len(df)]
df["Query Label"] = ["Query " + l for l in labels]

sns.set(style="whitegrid", font_scale=0.7)
plt.rcParams.update({'font.family': 'serif'})

fig_size = (4, 3)  # smaller for research paper

def save_barplot(x_col, palette, title, xlabel, filename):
    plt.figure(figsize=fig_size)
    sns.barplot(x=x_col, y="Query Label", data=df, palette=palette)
    plt.title(title, fontsize=10, fontweight='bold')
    plt.xlabel(xlabel, fontsize=8)
    plt.ylabel("Query", fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

save_barplot("Response Time (s)", "Blues_r", "Response Time per Query", "Seconds", "response_time.png")
save_barplot("Relevance Score", "Greens_r", "Relevance Score per Query", "Score (0-1)", "relevance_score.png")
save_barplot("Cosine Similarity", "Purples_r", "Cosine Similarity per Query", "Similarity (0-1)", "cosine_similarity.png")
save_barplot("Response Length", "Oranges_r", "Response Length per Query", "Number of Words", "response_length.png")

# Print legend (query labels with actual queries)
legend_df = df[["Query Label", "Query"]]
print("\nLegend (Query Label → Actual Query):")
print(legend_df.to_string(index=False))

# Optional: Zip all images into one file for easy download
with zipfile.ZipFile('query_analysis_plots.zip', 'w') as z:
    for f in ["response_time.png", "relevance_score.png", "cosine_similarity.png", "response_length.png"]:
        z.write(f)

print("\nAll plots saved as PNG and zipped in 'query_analysis_plots.zip'")
