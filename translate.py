import pandas as pd
from deep_translator import GoogleTranslator
import time

translator = GoogleTranslator(source='en', target='pl')

input_file = 'train.csv'
output_file = 'translated_train.csv'
test_file = 'test.csv'

df = pd.read_csv(input_file)

test_df = df.iloc[0:1]

test_df.to_csv(test_file, index=False)

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
            time.sleep(1)
    print(f"Failed to translate text '{text}' after {retries} retries.")
    return text

def translate_row(row):
    row['TITLE'] = translate_text(row['TITLE'])
    row['ABSTRACT'] = translate_text(row['ABSTRACT'])
    return row

translated_test_df = df.apply(translate_row, axis=1)

translated_test_df.to_csv(output_file, index=False)

print(f"Translation completed. Translated test dataset saved to '{output_file}'")

translated_test_df.to_csv(output_file, index=False)

print(f"Translation completed. Translated test dataset saved to '{output_file}'")
#