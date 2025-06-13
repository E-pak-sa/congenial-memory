import os
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import argparse
import re

def extract_texts_from_xml(xml_path):
    texts = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for text_elem in root.findall('.//TEXT'):
            if text_elem.text:
                text = text_elem.text.strip()
                if text:
                    texts.append(text)
    except Exception as e:
        print(f"[XML ERROR] {xml_path}: {e}")
    return texts

def extract_tweets_from_csv(csv_path):
    tweets = []
    try:
        df = pd.read_csv(csv_path)
        if 'tweet' in df.columns:
            tweets = df['tweet'].dropna().astype(str).tolist()
        else:
            print(f"[CSV WARNING] No 'tweet' column in {csv_path}")
    except Exception as e:
        print(f"[CSV ERROR] {csv_path}: {e}")
    return tweets

def remove_urls(text):
    return re.sub(r'(https?://\S+|www\.\S+)', '', text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reddit', type=int, default=None, help='Number of XML files to process per folder (pos/neg)')
    parser.add_argument('--tweet', type=int, default=None, help='Number of CSV files to process per folder (pos/neg)')
    args = parser.parse_args()

    root_dir = Path("Dataset_saving/train")
    save_dir = Path("Dataset_saving")
    output_csv = save_dir/ "merged_data.csv"
    rows = []
    for label_folder, label in [("pos", 1), ("neg", 0)]:
        folder = root_dir / label_folder
        files = [f for f in folder.rglob("*") if f.is_file()]
        # --m: 앞 m개 .csv만
        if args.tweet is not None:
            csv_files = [f for f in files if f.suffix.lower() == ".csv"][:args.tweet]
            for file_path in csv_files:
                tweets = extract_tweets_from_csv(file_path)
                for tweet in tweets:
                    filtered_text = remove_urls(tweet)
                    if len(filtered_text.strip()) >= 30:
                        rows.append({
                            "label": label,
                            "text": filtered_text,
                            "origin_file": str(file_path)
                        })
        # --n: 앞 n개 .xml만
        if args.reddit is not None:
            xml_files = [f for f in files if f.suffix.lower() == ".xml"][:args.reddit]
            for file_path in xml_files:
                texts = extract_texts_from_xml(file_path)
                for text in texts:
                    filtered_text = remove_urls(text)
                    if len(filtered_text.strip()) >= 40:
                        rows.append({
                            "label": label,
                            "text": filtered_text,
                            "origin_file": str(file_path)
                        })
        if args.reddit is None and args.tweet is None:
        # 기본: 모든 파일 처리
            for file_path in files:
                if file_path.suffix.lower() == ".xml":
                    texts = extract_texts_from_xml(file_path)
                    for text in texts:
                        filtered_text = remove_urls(text)
                        if len(filtered_text.strip()) >= 30:
                            rows.append({
                                "label": label,
                                "text": filtered_text,
                                "origin_file": str(file_path)
                            })
                elif file_path.suffix.lower() == ".csv":
                    tweets = extract_tweets_from_csv(file_path)
                    for tweet in tweets:
                        filtered_text = remove_urls(tweet)
                        if len(filtered_text.strip()) >= 30:
                            rows.append({
                                "label": label,
                                "text": filtered_text,
                                "origin_file": str(file_path)
                            })
    # Write to CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["label", "text", "origin_file"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Merged {len(rows)} rows into {output_csv}")

if __name__ == "__main__":
    main() 
