import pandas as pd

# Load the Human_Auras CSV file
human_auras = pd.read_csv("Human_Auras.csv")

# Calculate the sum of aura values
total_aura = human_auras["Aura"].sum()

# Calculate the total number of humans
total_humans = human_auras.shape[0]

# Calculate the average aura
average_aura = total_aura / total_humans

# Print the average aura rounded to 2 decimal places
print("Average Aura:", round(average_aura, 2))
