import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from collections import Counter
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch

# Load the Excel file
file_path = 'Normal_RAG_Request_Response.xlsx'
df = pd.read_excel(file_path)

# Display the columns in the DataFrame
print("Columns in the DataFrame:", df.columns)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def calculate_similarity(actual_response, expected_response):
    vectorizer = TfidfVectorizer()
    combined = [actual_response, expected_response]
    tfidf_matrix = vectorizer.fit_transform(combined)
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity_matrix[0][0]

def calculate_bleu_score(actual_response, expected_response):
    bleu_score = sentence_bleu(expected_response.split(), actual_response.split())
    return bleu_score

def calculate_rouge_score(actual_response, expected_response):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_score = scorer.score(expected_response, actual_response)
    return rouge_score

def calculate_f1_score(actual_response, expected_response):
    result = f1_score(expected_response, actual_response)
    return result

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def f1_score(reference, generated):
    # Tokenize reference and generated answers
    ref_tokens = tokenize(reference)
    gen_tokens = tokenize(generated)

    # Count occurrences of each token
    ref_counts = Counter(ref_tokens)
    gen_counts = Counter(gen_tokens)

    # Calculate true positives (TP), false positives (FP), and false negatives (FN)
    tp = sum(min(ref_counts[token], gen_counts[token]) for token in ref_counts)
    fp = sum(gen_counts[token] - ref_counts[token] for token in gen_counts if token not in ref_counts)
    fn = sum(ref_counts[token] - gen_counts[token] for token in ref_counts if token not in gen_counts)

    # Precision, Recall, and F1 Score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1

def calculate_bert_similarity(actual_response, expected_response):
    # Tokenize input texts
    inputs = tokenizer([actual_response, expected_response], return_tensors='pt', padding=True, truncation=True)
    
    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling over tokens
    
    # Calculate cosine similarity
    similarity = cosine_similarity(embeddings[0].unsqueeze(0).numpy(), embeddings[1].unsqueeze(0).numpy())
    return similarity[0][0]

# Similarity Calculation
df['Similarity Score'] = df.apply(
    lambda row: calculate_similarity(
        row['Actual Response'],
        row['Expected Response']),
    axis=1)

# Bleu Score Calculation
df['Bleu_score'] = df.apply(
    lambda row: calculate_bleu_score(
        row['Actual Response'],
        row['Expected Response']),
    axis=1)

# ROUGE score calculation
df['Rouge1_precision'] = df.apply(
    lambda row: calculate_rouge_score(
        row['Actual Response'],
        row['Expected Response']).get('rouge1').precision,
    axis=1)

df['Rouge1_recall'] = df.apply(
    lambda row: calculate_rouge_score(
        row['Actual Response'],
        row['Expected Response']).get('rouge1').recall,
    axis=1)

df['Rouge1_fmeasure'] = df.apply(
    lambda row: calculate_rouge_score(
        row['Actual Response'],
        row['Expected Response']).get('rouge1').fmeasure,
    axis=1)

df['RougeL_precision'] = df.apply(
    lambda row: calculate_rouge_score(
        row['Actual Response'],
        row['Expected Response']).get('rougeL').precision,
    axis=1)

df['RougeL_recall'] = df.apply(
    lambda row: calculate_rouge_score(
        row['Actual Response'],
        row['Expected Response']).get('rougeL').recall,
    axis=1)

df['RougeL_fmeasure'] = df.apply(
    lambda row: calculate_rouge_score(
        row['Actual Response'],
        row['Expected Response']).get('rougeL').fmeasure,
    axis=1)

# F1 score calculation
df['F1_score'] = df.apply(
    lambda row: calculate_f1_score(
        row['Actual Response'],
        row['Expected Response']),
    axis=1)

# BERT similarity calculation
df['BERT_Similarity'] = df.apply(
    lambda row: calculate_bert_similarity(
        row['Actual Response'],
        row['Expected Response']),
    axis=1)

# Display the updated DataFrame with Bleu_score
print("DataFrame with Bleu_score, Rouge1_precision, F1_score, and BERT_Similarity")
print(df[['Question', 'Bleu_score', 'Rouge1_precision', 'F1_score', 'BERT_Similarity']])

# Save the results to a new Excel file
output_file_path = 'Result_Normal_RAG_Request_Response.xlsx'  # Change this to your desired output path
df.to_excel(output_file_path, index=False)
print(f"Results saved to {output_file_path}")