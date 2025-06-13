import nltk
from collections import Counter
import spacy
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import json
import argparse

# Download required NLTK data files if not already present
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nlp = spacy.load('en_core_web_sm')

FEATURES = [
    'single_first_person',
    'plural_first_person',
    'third_person',
    'conjunctions',
    'auxiliary_verbs',
    'prepositions'
]

def analyze_text_features(text, method='nltk'):
    """
    Analyze a text string and return a dictionary with counts for:
    (1) number of single first person pronouns
    (2) number of plural first person pronouns
    (3) number of third person pronouns
    (4) number of conjunctions
    (5) number of auxiliary verbs
    (6) number of prepositions
    method: 'nltk' (default) or 'spacy'
    """
    if method == 'spacy':
        doc = nlp(text)
        counts = Counter()
        for token in doc:
            # Pronouns
            if token.pos_ == 'PRON':
                lw = token.text.lower()
                # First person singular
                if lw in ("i", "me", "my", "mine", "myself"):
                    counts['single_first_person'] += 1
                # First person plural
                elif lw in ("we", "us", "our", "ours", "ourselves"):
                    counts['plural_first_person'] += 1
                # Third person
                elif lw in ("he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves"):
                    counts['third_person'] += 1
            # Conjunctions
            if token.pos_ == 'CCONJ':
                counts['conjunctions'] += 1
            # Auxiliary verbs
            if token.tag_ in ('MD',) or token.dep_ == 'aux':
                counts['auxiliary_verbs'] += 1
            # Prepositions
            if token.pos_ == 'ADP':
                counts['prepositions'] += 1
        return {
            'single_first_person': counts['single_first_person'],
            'plural_first_person': counts['plural_first_person'],
            'third_person': counts['third_person'],
            'conjunctions': counts['conjunctions'],
            'auxiliary_verbs': counts['auxiliary_verbs'],
            'prepositions': counts['prepositions']
        }
    else:
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        
        counts = Counter()
        for word, tag in pos_tags:
            # Pronouns
            if tag in ("PRP", "PRP$"):
                lw = word.lower()
                # First person singular
                if lw in ("i", "me", "my", "mine", "myself"):
                    counts['single_first_person'] += 1
                # First person plural
                elif lw in ("we", "us", "our", "ours", "ourselves"):
                    counts['plural_first_person'] += 1
                # Third person
                elif lw in ("he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves"):
                    counts['third_person'] += 1
            # Conjunctions
            if tag == "CC":
                counts['conjunctions'] += 1
            # Auxiliary verbs (modals)
            if tag == "MD":
                counts['auxiliary_verbs'] += 1
            # Prepositions
            if tag == "IN":
                counts['prepositions'] += 1
        return {
            'single_first_person': counts['single_first_person'],
            'plural_first_person': counts['plural_first_person'],
            'third_person': counts['third_person'],
            'conjunctions': counts['conjunctions'],
            'auxiliary_verbs': counts['auxiliary_verbs'],
            'prepositions': counts['prepositions']
        } 

def analyze_features_by_group(df, group_col, method='nltk'):
    """
    For a DataFrame with a 'text' column and a group column (e.g., 'group'),
    apply analyze_text_features to each row's text, and return a dict with
    the mean and std for each feature for each group.
    Returns: {group1: {feature: {'mean': x, 'std': y}, ...}, group2: ...}
    """
    # Store results for each row
    feature_dicts = df['text'].apply(lambda x: analyze_text_features(x, method=method))
    features_df = pd.DataFrame(list(feature_dicts))
    combined = pd.concat([df[[group_col]].reset_index(drop=True), features_df], axis=1)
    result = {}
    for group, group_df in combined.groupby(group_col):
        stats = {}
        for col in features_df.columns:
            stats[col] = {
                'mean': group_df[col].mean(),
                'std': group_df[col].std()
            }
        result[group] = stats
    return result
    
def cohen_d(x, y):
    """Compute Cohen's d for two arrays."""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else np.nan

def t_test_pvalue(x, y):
    """Compute t-test p-value for two arrays."""
    t_stat, p_val = ttest_ind(x, y, equal_var=False, nan_policy='omit')
    return p_val


def analyze_by_topic_and_label(df, method='nltk'):
    """
    For a large DataFrame with 'topic_label' and 'label' columns:
    1. Split by 'topic_label' (numeric)
    2. For each topic_label, split by 'label' (0/1), print counts
    3. For each split, run analyze_features_by_group
    4. For each feature, compute Cohen's d and t-test p-value between label 0 and 1
    5. Print or return all results
    """
    results = {}
    topic_labels = df['topic_label'].unique()
    for topic in topic_labels:
        topic_df = df[df['topic_label'] == topic]
        label0_df = topic_df[topic_df['label'] == 0]
        label1_df = topic_df[topic_df['label'] == 1]
        print(f"Topic {topic}: label=0 count={len(label0_df)}, label=1 count={len(label1_df)}")
        if len(label0_df) == 0 or len(label1_df) == 0:
            print(f"  Skipping topic {topic} due to missing label group.")
            continue
        # Analyze features for each group
        group_stats = analyze_features_by_group(topic_df, group_col='label', method=method)
        # For each feature, compute Cohen's d and t-test
        feature_results = {}
        for feature in group_stats[0].keys():
            x = label0_df['text'].apply(lambda t: analyze_text_features(t, method=method)[feature])
            y = label1_df['text'].apply(lambda t: analyze_text_features(t, method=method)[feature])
            d = cohen_d(x, y)
            p = t_test_pvalue(x, y)
            feature_results[feature] = {
                'label0_mean': group_stats[0][feature]['mean'],
                'label0_std': group_stats[0][feature]['std'],
                'label1_mean': group_stats[1][feature]['mean'],
                'label1_std': group_stats[1][feature]['std'],
                "cohen_d": d,
                "t_test_p": p
            }
        results[topic] = feature_results
        print(f"  Feature stats for topic {topic}:")
        for feat, vals in feature_results.items():
            print(f"    {feat}: {vals}")
    return results
    
def save_results_to_json(results, filepath):
    """Save the results dictionary to a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
def show_top_topics_and_examples(results, df, feature, n=3, pval_threshold=0.05):
    """
    Show top n topics for a given feature, sorted by Cohen's d (abs), filtered by p-value.
    For each topic, print one example sentence for label 0 and label 1.
    """
    topic_stats = []
    for topic, feats in results.items():
        stat = feats[feature]
        topic_stats.append({
            'topic': topic,
            'cohen_d': abs(stat['cohen_d']),
            'pval': stat['t_test_p']
        })
    filtered = [t for t in topic_stats if t['pval'] < pval_threshold]
    sorted_topics = sorted(filtered, key=lambda x: x['cohen_d'], reverse=True)[:n]
    print(f"\nTop {n} topics for feature '{feature}' (p<{pval_threshold}):")
    for t in sorted_topics:
        topic = t['topic']
        print(f"Topic {topic}: Cohen's d={t['cohen_d']:.3f}, p={t['pval']:.4g}")
        for label in [0, 1]:
            ex = df[(df['topic_label'] == topic) & (df['label'] == label)]['text']
            ex_text = ex.iloc[0] if len(ex) > 0 else '[No example]'
            print(f"  label={label} example: {ex_text}")

def show_top_topics_for_all_features(results, df, n=3, pval_threshold=0.05):
    """
    For all FEATURES, show top n topics and example sentences.
    """
    for feature in FEATURES:
        show_top_topics_and_examples(results, df, feature, n=n, pval_threshold=pval_threshold)

def main():
    parser = argparse.ArgumentParser(description='Analyze linguistic features by topic and label.')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file (must have columns: text, topic_label, label)')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON file')
    parser.add_argument('--method', type=str, default='nltk', choices=['nltk', 'spacy'], help='Which NLP method to use (nltk or spacy)')
    parser.add_argument('--n', type=int, default=3, help='Number of top topics to show per feature')
    parser.add_argument('--pval_threshold', type=float, default=0.05, help='p-value threshold for significance')
    parser.add_argument('--feature', type=str, default='all', help="Feature to show top topics for (or 'all')")
    args = parser.parse_args()

    print(f"Loading data from {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows.")

    print(f"Analyzing features by topic and label using {args.method} ...")
    results = analyze_by_topic_and_label(df, method=args.method)

    print(f"Saving results to {args.output} ...")
    save_results_to_json(results, args.output)
    print("Done.")

    # Show top topics for selected feature(s)
    if args.feature == 'all':
        show_top_topics_for_all_features(results, df, n=args.n, pval_threshold=args.pval_threshold)
    else:
        if args.feature not in FEATURES:
            print(f"Feature '{args.feature}' not recognized. Available: {FEATURES}")
        else:
            show_top_topics_and_examples(results, df, feature=args.feature, n=args.n, pval_threshold=args.pval_threshold)

if __name__ == '__main__':
    main()
    
    
