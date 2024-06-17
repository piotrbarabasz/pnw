import pandas as pd
from deep_translator import GoogleTranslator
import time

# Initialize the translator
translator = GoogleTranslator(source='en', target='pl')

# Load the CSV file
input_file = 'train.csv'
output_file = 'translated_train.csv'
test_file = 'test.csv'
# Read the entire CSV file
df = pd.read_csv(input_file)

# Create a test dataset with rows from 2nd to 12th
test_df = df.iloc[0:1]

# Save the test dataset to a new CSV file
test_df.to_csv(test_file, index=False)

# Function to translate text
def translate_text(text, retries=3):
    if pd.isna(text):
        return text
    attempt = 0
    while attempt < retries:
        try:
            translated = translator.translate(text)
            print(f"Original: {text}\nTranslated: {translated}")
            return translated
        except Exception as e:
            print(f"Error translating text '{text}': {e}. Retrying...")
            attempt += 1
            time.sleep(1)  # Wait for 1 second before retrying
    print(f"Failed to translate text '{text}' after {retries} retries.")
    return text

# Function to translate specific columns in a row
def translate_row(row):
    row['TITLE'] = translate_text(row['TITLE'])
    row['ABSTRACT'] = translate_text(row['ABSTRACT'])
    return row

# Apply translation to the test dataset
translated_test_df = df.apply(translate_row, axis=1)

# Save the translated test dataset to a new CSV file
translated_test_df.to_csv(output_file, index=False)

print(f"Translation completed. Translated test dataset saved to '{output_file}'")

# Save the translated test dataset to a new CSV file
translated_test_df.to_csv(output_file, index=False)

print(f"Translation completed. Translated test dataset saved to '{output_file}'")
#
# # Read the CSV file in chunks
# chunk_size = 100  # You can adjust this based on memory constraints
# chunks = pd.read_csv(input_file, chunksize=chunk_size)
#
# def translate_text(text):
#     if pd.isna(text):
#         return text
#     try:
#         translated = translator.translate(text, src='en', dest='pl').text
#         return translated
#     except Exception as e:
#         print(f"Error translating text '{text}': {e}")
#         return text
#
# # Function to translate a whole row
# def translate_row(row):
#     return row.map(translate_text)
#
# # Iterate over chunks and translate
# for i, chunk in enumerate(chunks):
#     translated_chunk = chunk.apply(translate_row, axis=1)
#     if i == 0:
#         # Write the header only for the first chunk
#         translated_chunk.to_csv(output_file, index=False, mode='w', header=True)
#     else:
#         # Append without writing the header
#         translated_chunk.to_csv(output_file, index=False, mode='a', header=False)
#     print(f"Processed chunk {i+1}")
#
# print(f"Translation completed. Translated dataset saved to '{output_file}'")