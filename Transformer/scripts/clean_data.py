import sys
import os
import shutil
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data.preprocess import TextPreprocessor

def main():
    raw_dir = "data/raw"
    clean_dir = "data/clean"
    os.makedirs(clean_dir, exist_ok=True)
    
    preprocessor = TextPreprocessor()
    
    files = [
        ("train.en", "en"), 
        ("train.vi", "vi"),
        ("dev.en", "en"),
        ("dev.vi", "vi"),
        ("test.en", "en"),
        ("test.vi", "vi")
    ]
    
    print("--- Starting Data Cleaning ---")
    
    for filename, lang in files:
        src_path = os.path.join(raw_dir, filename)
        tgt_path = os.path.join(clean_dir, filename)
        
        if not os.path.exists(src_path):
            print(f"Skipping {filename} (Not found)")
            continue
            
        print(f"Processing {filename} as {lang}...")
        
        with open(src_path, 'r', encoding='utf-8') as f_in, \
             open(tgt_path, 'w', encoding='utf-8') as f_out:
            
            lines = f_in.readlines()
            for line in tqdm(lines):
                cleaned = preprocessor.process(line, lang)
                f_out.write(cleaned + "\n")
                
    print(f"--- Completed. Clean data saved to {clean_dir} ---")
    
    # Auto-clear cache
    cache_dir = "data/cache"
    if os.path.exists(cache_dir):
        print(f"Clearing old cache at {cache_dir}...")
        shutil.rmtree(cache_dir)
        print("Cache cleared.")

if __name__ == "__main__":
    main()
