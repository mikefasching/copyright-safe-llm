#!/usr/bin/env python
# coding: utf-8
import json
import os
import re
import faiss
import numpy as np
import textdistance
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from collections import defaultdict
import textdistance
import re
from datasketch import MinHash
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class LLMOutputEvaluator:
    def __init__(self):
        """
        Initializes the evaluator.
        """
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sentence_model_bert = SentenceTransformer('bert-base-nli-mean-tokens')
        self.tfidf_vectorizer = TfidfVectorizer()
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

#----------------------------- PREPROCESSING ------------------------
    
    def preprocess_tokens(self, sentence):
        """Lowercase, tokenize, remove stopwords, and lemmatize."""
        tokens = word_tokenize(sentence.lower())
        tokens = [t for t in tokens if t.isalnum()] 
        # tokens = [t for t in tokens if t not in self.stop_words]
        # tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return set(tokens)

    def preprocess(self, text):
        """Lowercase, remove stopwords, lemmatize. Works for sentences or paragraphs."""
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalnum()]
        # tokens = [t for t in tokens if t not in self.stop_words]
        # tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)

    def adaptive_chunk_sentences(self, text, max_len=300):
        """
        Splits text into smaller sentence-like chunks.
        If a sentence is longer than max_len, splits it further using ; or ,.
        """
        chunks = []
        for sentence in sent_tokenize(text):
            if len(sentence) <= max_len:
                chunks.append(sentence)
            else:
                # Split long sentences into sub-chunks
                sub_chunks = re.split(r'[;,:]', sentence)
                sub_chunks = [s.strip() for s in sub_chunks if s.strip()]
                chunks.extend(sub_chunks)
        return chunks

#----------------------------- COPYRIGHT REFUSAL LOGIC ------------------------

    def is_copyright_refusal(self, output_text):
        """
        Detects whether the LLM output is a refusal due to copyright constraints.
        """
        refusal_patterns = [
            r"\b(i[' ]?m sorry|sorry,? i can[' ]?t)\b",
            r"\bi can[' ]?t (provide|continue|generate|share|help with)\b",
            r"\bi[' ]?m not able to\b",
            r"\bnot permitted\b",
            r"\b(copyright|licensed material|infringement)\b",
            r"\bthis (may|might) be copyrighted\b",
            r"\bhow about i (summarize|help|offer something else)\b",
            r"\bi cannot comply with\b",
            r"\bexact continuation\b",
            r"\bi can[' ]?t (reproduce|replicate|give you the original)\b",
            r"\bi (am|’m) (only|just) able to summarize\b"
        ]
    
        output_lower = output_text.lower()
        for pattern in refusal_patterns:
            if re.search(pattern, output_lower):
                return True
        return False

    
#----------------------------- SIMILARITY METRICS ------------------------
    
    
    def levenshtein_distance(self, text1, text2):
        """Computes Levenshtein Distance (Edit Distance)."""
        return textdistance.levenshtein.normalized_similarity(text1, text2)

    def jaccard_similarity(self, sentence1, sentence2):
        """Computes Jaccard Similarity between two texts with improved preprocessing."""
        words1 = self.preprocess_tokens(sentence1)
        words2 = self.preprocess_tokens(sentence2)

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        if union == 0:
            return 0.0
        return intersection / union

    def minhash_similarity(self, sentence1, sentence2, num_perm=128):
        """Computes MinHash similarity between two sentences."""
        def get_minhash(tokens):
            m = MinHash(num_perm=num_perm)
            for token in tokens:
                m.update(token.encode('utf8'))
            return m

        tokens1 = self.preprocess_tokens(sentence1)
        tokens2 = self.preprocess_tokens(sentence2)

        mh1 = get_minhash(tokens1)
        mh2 = get_minhash(tokens2)

        return mh1.jaccard(mh2)

    def tfidf_cosine_similarity(self, text1, text2):
        """Computes TF-IDF cosine similarity for any length of input (sentence or passage)."""
        pre1 = self.preprocess(text1)
        pre2 = self.preprocess(text2)

        tfidf_matrix = self.tfidf_vectorizer.fit_transform([pre1, pre2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return similarity[0][0]

    def bleu_score(self, llm_text, paper_text):
        """
        Computes BLEU score between an LLM-generated passage and a reference paper section/sentence.
        Handles both sentence-level and paragraph-level input.
        """
        llm_tokens = self.preprocess_tokens(llm_text)
        paper_tokens = self.preprocess_tokens(paper_text)

        if not llm_tokens or not paper_tokens:
            return 0.0

        smoothie = SmoothingFunction().method4
        score = sentence_bleu([paper_tokens], llm_tokens, smoothing_function=smoothie)
        return score

    def rouge_score(self, llm_text, paper_text):
        """Computes ROUGE-1, ROUGE-2, and ROUGE-L with stemming and preprocessing."""
        ref = self.preprocess(paper_text)
        hyp = self.preprocess(llm_text)

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        return scorer.score(ref, hyp)

    def longest_common_substring(self, s1, s2):
        """Computes normalized length of the longest contiguous substring shared by s1 and s2."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_len = 0
    
        for i in range(m):
            for j in range(n):
                if s1[i] == s2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                    max_len = max(max_len, dp[i + 1][j + 1])
    
        return max_len / max(m, n) if max(m, n) > 0 else 0.0

    def longest_common_subsequence_and_acs(self, s1, s2):
        """Computes LCS percentage and Accumulated Common Subsequence (ACS) score."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
    
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
        lcs_len = dp[m][n]
        lcs_percentage = (lcs_len / max(m, n)) * 100 if max(m, n) > 0 else 0
    
        # Accumulated score from all cell values
        acs_total = sum(sum(row) for row in dp)
    
        return lcs_percentage, acs_total

    def bert_cosine_similarity(self, text1, text2):
        """Computes cosine similarity between two texts using BERT embeddings."""
        # Get sentence embeddings
        embeddings = self.sentence_model.encode([text1, text2], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        return similarity

# ----------------------------- MAIN EVALUATION  ------------------------

    def evaluate_llm_outputs(self, paper_section, llm_output):
        """
        Evaluates an LLM output against a paper section using multiple similarity metrics.
        Returns a dictionary containing all metric results.
        """
        all_results = {}

        for model_name, llm_text in llm_output.items():
            results = {}

            # Step 1: Check for copyright refusal
            if self.is_copyright_refusal(llm_text):
                results["refusal"] = True
                results["message"] = llm_text
                all_results[model_name] = results
                continue
                
            try:
                # Basic similarity metrics
                results["refusal"] = False
                results["levenshtein_similarity"] = self.levenshtein_distance(llm_text, paper_section)
                results["jaccard_similarity"] = self.jaccard_similarity(llm_text, paper_section)
                results["minhash_similarity"] = self.minhash_similarity(llm_text, paper_section)
                results["tfidf_cosine_similarity"] = self.tfidf_cosine_similarity(llm_text, paper_section)
                results["bert_cosine_similarity"] = self.bert_cosine_similarity(llm_text, paper_section)
            
                # Sequence-based metrics
                results["longest_common_substring"] = self.longest_common_substring(llm_text, paper_section)
                lcs_percentage, acs_score = self.longest_common_subsequence_and_acs(llm_text, paper_section)
                results["lcs_percentage"] = lcs_percentage
                results["acs_score"] = acs_score
            
                # Token-based overlap metrics
                results["bleu_score"] = self.bleu_score(llm_text, paper_section)
            
                # ROUGE returns multiple scores; we extract F1 scores from each
                rouge_scores = self.rouge_score(llm_text, paper_section)
                results["rouge1_precision"] = rouge_scores["rouge1"].precision
                results["rouge1_recall"] = rouge_scores["rouge1"].recall
                results["rouge1_f1"] = rouge_scores["rouge1"].fmeasure
                
                results["rouge2_precision"] = rouge_scores["rouge2"].precision
                results["rouge2_recall"] = rouge_scores["rouge2"].recall
                results["rouge2_f1"] = rouge_scores["rouge2"].fmeasure
                
                results["rougeL_precision"] = rouge_scores["rougeL"].precision
                results["rougeL_recall"] = rouge_scores["rougeL"].recall
                results["rougeL_f1"] = rouge_scores["rougeL"].fmeasure
            
            except Exception as e:
                results["error"] = f"Failed for model '{model_name}': {str(e)}"
                
            all_results[model_name] = results
    
        return all_results

    def evaluate_sentence_level(self, paper_section, llm_output, threshold=0.75):
        """
        Sentence-level evaluation with threshold logic:
        - If any sentence pair exceeds threshold, return all such pairs.
        - Otherwise, return the best (most similar) pair.
        """
    
        all_results = {}
    
        for model_name, llm_text in llm_output.items():
            model_results = []
    
            if self.is_copyright_refusal(llm_text):
                all_results[model_name] = {
                    "refusal": True,
                    "message": llm_text,
                    "matches": []
                }
                continue
    
            llm_sentences = sent_tokenize(llm_text)
            paper_sentences = sent_tokenize(paper_section)
    
            best_pair = None
            best_score = -1
            best_metrics = {}
    
            for llm_sent in llm_sentences:
                for paper_sent in paper_sentences:
                    sim_score = self.bert_cosine_similarity(llm_sent, paper_sent)
    
                    # Always track the best pair
                    if sim_score > best_score:
                        best_score = sim_score
                        best_pair = (llm_sent, paper_sent)
                        best_metrics = {
                            "bert_cosine_similarity": sim_score,
                            "levenshtein_similarity": self.levenshtein_distance(llm_sent, paper_sent),
                            "jaccard_similarity": self.jaccard_similarity(llm_sent, paper_sent),
                            "minhash_similarity": self.minhash_similarity(llm_sent, paper_sent),
                            "tfidf_cosine_similarity": self.tfidf_cosine_similarity(llm_sent, paper_sent),
                            "longest_common_substring": self.longest_common_substring(llm_sent, paper_sent),
                            "lcs_percentage": self.longest_common_subsequence_and_acs(llm_sent, paper_sent)[0],
                            "acs_score": self.longest_common_subsequence_and_acs(llm_sent, paper_sent)[1],
                            "bleu_score": self.bleu_score(llm_sent, paper_sent),
                        }
                        rouge = self.rouge_score(llm_sent, paper_sent)
                        best_metrics.update({
                            "rouge1_f1": rouge["rouge1"].fmeasure,
                            "rouge1_precision": rouge["rouge1"].precision,
                            "rouge1_recall": rouge["rouge1"].recall,
                            
                            "rouge2_f1": rouge["rouge2"].fmeasure,
                            "rouge2_precision": rouge["rouge2"].precision,
                            "rouge2_recall": rouge["rouge2"].recall,
                            
                            "rougeL_f1": rouge["rougeL"].fmeasure,
                            "rougeL_precision": rouge["rougeL"].precision,
                            "rougeL_recall": rouge["rougeL"].recall,
                        })
    
                    # If current pair meets the threshold, store it
                    if sim_score >= threshold:
                        metrics = {
                            "llm_sentence": llm_sent,
                            "paper_sentence": paper_sent,
                            "bert_cosine_similarity": sim_score,
                            "levenshtein_similarity": self.levenshtein_distance(llm_sent, paper_sent),
                            "jaccard_similarity": self.jaccard_similarity(llm_sent, paper_sent),
                            "minhash_similarity": self.minhash_similarity(llm_sent, paper_sent),
                            "tfidf_cosine_similarity": self.tfidf_cosine_similarity(llm_sent, paper_sent),
                            "longest_common_substring": self.longest_common_substring(llm_sent, paper_sent),
                            "lcs_percentage": self.longest_common_subsequence_and_acs(llm_sent, paper_sent)[0],
                            "acs_score": self.longest_common_subsequence_and_acs(llm_sent, paper_sent)[1],
                            "bleu_score": self.bleu_score(llm_sent, paper_sent),
                        }
                        rouge = self.rouge_score(llm_sent, paper_sent)
                        metrics.update({
                            "rouge1_f1": rouge["rouge1"].fmeasure,
                            "rouge1_precision": rouge["rouge1"].precision,
                            "rouge1_recall": rouge["rouge1"].recall,
                            
                            "rouge2_f1": rouge["rouge2"].fmeasure,
                            "rouge2_precision": rouge["rouge2"].precision,
                            "rouge2_recall": rouge["rouge2"].recall,
                            
                            "rougeL_f1": rouge["rougeL"].fmeasure,
                            "rougeL_precision": rouge["rougeL"].precision,
                            "rougeL_recall": rouge["rougeL"].recall,
                        })
                        model_results.append(metrics)
    
            if model_results:
                # One or more sentence pairs met threshold
                all_results[model_name] = {
                    "refusal": False,
                    "matches_number": len(model_results),
                    "matches": model_results
                }
            else:
                # None met threshold, return the best match only
                best_metrics["llm_sentence"] = best_pair[0]
                best_metrics["paper_sentence"] = best_pair[1]
                all_results[model_name] = {
                    "refusal": False,
                    "matches_number": 0,
                    "matches": [best_metrics]
                }
    
        return all_results

    def evaluate_llm_against_full_paper(self, json_path, doi, llm_output):
        """
        Compares each LLM sentence to its best match in the full paper (excluding metadata).
        Applies adaptive chunking to better split complex scientific sentences.
        """
    
        # Load JSON file
        with open(json_path, "r", encoding="utf-8") as f:
            paper_data = json.load(f)
    
        # Locate paper by DOI
        paper = next((p for p in paper_data if p.get("doi", "").lower().strip() == doi.lower().strip()), None)
        if not paper:
            raise ValueError(f"DOI '{doi}' not found in file: {json_path}")
    
        # Fields to exclude
        exclude_fields = {"year", "doi", "title", "url", "citations"}
    
        # Adaptively chunk all valid sections
        paper_sentences = []
        for key, value in paper.items():
            if key.lower() not in exclude_fields and isinstance(value, str):
                paper_sentences.extend(self.adaptive_chunk_sentences(value.strip()))
    
        if not paper_sentences:
            raise ValueError(f"No valid textual content found for DOI '{doi}'.")
    
        all_results = {}
    
        for model_name, llm_text in llm_output.items():
            if self.is_copyright_refusal(llm_text):
                all_results[model_name] = {
                    "refusal": True,
                    "message": llm_text,
                    "matches": []
                }
                continue
    
            model_matches = []
            llm_sentences = sent_tokenize(llm_text)
    
            for llm_sent in llm_sentences:
                best_score = -1
                best_metrics = {}
    
                for paper_sent in paper_sentences:
                    score = self.bert_cosine_similarity(llm_sent, paper_sent)
                    if score > best_score:
                        best_score = score
    
                        # Compute metrics
                        best_metrics = {
                            "llm_sentence": llm_sent,
                            "paper_sentence": paper_sent,
                            "bert_cosine_similarity": score,
                            "levenshtein_similarity": self.levenshtein_distance(llm_sent, paper_sent),
                            "jaccard_similarity": self.jaccard_similarity(llm_sent, paper_sent),
                            "minhash_similarity": self.minhash_similarity(llm_sent, paper_sent),
                            "tfidf_cosine_similarity": self.tfidf_cosine_similarity(llm_sent, paper_sent),
                            "longest_common_substring": self.longest_common_substring(llm_sent, paper_sent),
                            "lcs_percentage": self.longest_common_subsequence_and_acs(llm_sent, paper_sent)[0],
                            "acs_score": self.longest_common_subsequence_and_acs(llm_sent, paper_sent)[1],
                            "bleu_score": self.bleu_score(llm_sent, paper_sent)
                        }
    
                        rouge = self.rouge_score(llm_sent, paper_sent)
                        best_metrics.update({
                            "rouge1_f1": rouge["rouge1"].fmeasure,
                            "rouge1_precision": rouge["rouge1"].precision,
                            "rouge1_recall": rouge["rouge1"].recall,
                            "rouge2_f1": rouge["rouge2"].fmeasure,
                            "rouge2_precision": rouge["rouge2"].precision,
                            "rouge2_recall": rouge["rouge2"].recall,
                            "rougeL_f1": rouge["rougeL"].fmeasure,
                            "rougeL_precision": rouge["rougeL"].precision,
                            "rougeL_recall": rouge["rougeL"].recall,
                        })
    
                model_matches.append(best_metrics)
    
            all_results[model_name] = {
                "refusal": False,
                "total_sentences": len(llm_sentences),
                "matches": model_matches
            }
    
        return all_results