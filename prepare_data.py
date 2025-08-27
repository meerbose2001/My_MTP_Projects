import pandas as pd

# Load both files
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add label column
fake["label"] = "fake"
true["label"] = "real"

# Keep only the needed columns
fake = fake[["title", "text", "label"]]
true = true[["title", "text", "label"]]

# Combine
df = pd.concat([fake, true], ignore_index=True)

# Save to new CSV
df.to_csv("combined.csv", index=False)

print("combined.csv created successfully!")
print(df.head())
