import pandas as pd

# Load the Excel file
excel_file = 'mental_health_dataset.xlsx'  # Replace with your Excel file name
sheet_name = 'Sheet1'  # Replace with your sheet name or use None for the first sheet

# Read the Excel file
df = pd.read_excel(excel_file, sheet_name=sheet_name)

# Save to CSV with a semicolon as the delimiter
df.to_csv('output_file.csv', sep=';', index=False)  # Replace with your desired output name
