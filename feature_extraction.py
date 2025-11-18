
"""
Batch scraper for two datasets:
 - tranco_list.csv  -> structured_data_legitimate.csv (label=0)
 - verified_online.csv -> structured_data_phishing.csv (label=1)

Requirements:
 - requests
 - beautifulsoup4
 - pandas
 - feature_extraction.py (must expose create_vector(soup))
"""

import requests
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
from bs4 import BeautifulSoup
import pandas as pd
import feature_extraction as fe
import features as fe
import time
import os
from urllib.parse import urlparse, urlunparse


disable_warnings(InsecureRequestWarning)

# ----------------------------
# Config (edit these if needed)
# ----------------------------
BATCH_SIZE = 500        # number of URLs per batch
REQUEST_TIMEOUT = 6     # seconds per request
DELAY_BETWEEN_REQUESTS = 0.25  # polite delay (seconds)
# Optional slice limits (set to None for full file)
LEGIT_START = 1000
LEGIT_END = 2000      # only 100 URLs from tranco_list.csv
PHISH_START = 0
PHISH_END = 1000      # only 100 URLs from verified_online.csv
BATCH_SIZE = 100 

TRUSTED_LEGIT_INPUT = "tranco_list.csv"
TRUSTED_LEGIT_OUTPUT = "structured_data_legitimate.csv"

PHISH_INPUT = "verified_online.csv"
PHISH_OUTPUT = "structured_data_phishing.csv"

# ----------------------------
# Columns (must match feature_extraction.create_vector order + URL + label)
# ----------------------------
COLUMNS = [
    'has_title','has_input','has_button','has_image','has_submit','has_link',
    'has_password','has_email_input','has_hidden_element','has_audio','has_video',
    'number_of_inputs','number_of_buttons','number_of_images','number_of_option',
    'number_of_list','number_of_th','number_of_tr','number_of_href','number_of_paragraph',
    'number_of_script','length_of_title','has_h1','has_h2','has_h3','length_of_text',
    'number_of_clickable_button','number_of_a','number_of_img','number_of_div',
    'number_of_figure','has_footer','has_form','has_text_area','has_iframe','has_text_input',
    'number_of_meta','has_nav','has_object','has_picture','number_of_sources','number_of_span',
    'number_of_table','URL','label'
]

# ----------------------------
# Helpers: normalization & CSV loading
# ----------------------------
def normalize_url(raw):
    """Normalize a raw string into a full URL with scheme. Returns None if invalid."""
    if not isinstance(raw, str):
        return None
    u = raw.strip()
    if not u:
        return None
    # skip javascript/mailto/data URIs
    if u.lower().startswith(("mailto:", "javascript:", "data:")):
        return None
    # if it's already like http(s):// keep it
    if u.startswith("//"):
        u = "https:" + u
    parsed = urlparse(u)
    if not parsed.scheme:
        # add https by default
        u = "https://" + u
        parsed = urlparse(u)
    if not parsed.netloc:
        return None
    cleaned = urlunparse((parsed.scheme, parsed.netloc, parsed.path or "/", parsed.params, parsed.query, parsed.fragment))
    return cleaned

def load_urls_from_csv(path):
    """
    Robustly load URLs from CSV:
    - If CSV has column named 'url', uses that.
    - Else tries to use second column (index 1) if present.
    Returns list of normalized URLs (duplicates kept).
    """
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        # try without header
        df = pd.read_csv(path, header=None, dtype=str)
    # prefer explicit 'url' column if present (case-insensitive)
    cols_lower = [c.lower() for c in df.columns.astype(str)]
    if 'url' in cols_lower:
        # find actual column name
        actual = df.columns[cols_lower.index('url')]
        raws = df[actual].astype(str).tolist()
    else:
        # try second column (index 1), then first (index 0)
        if 1 in df.columns:
            raws = df.iloc[:, 1].astype(str).tolist()
        else:
            raws = df.iloc[:, 0].astype(str).tolist()
    urls = []
    for r in raws:
        n = normalize_url(r)
        if n:
            urls.append(n)
    return urls

# ----------------------------
# Scraping helper
# ----------------------------
def scrape_to_soup(url, timeout=REQUEST_TIMEOUT):
    try:
        r = requests.get(url, verify=False, timeout=timeout)
        if r.status_code == 200:
            return BeautifulSoup(r.content, "html.parser")
        else:
            print(f"[WARN] HTTP {r.status_code} for URL: {url}")
            return None
    except requests.RequestException as e:
        print(f"[ERROR] Request failed for URL {url}: {e}")
        return None

# ----------------------------
# Save helper (append, add header only if file missing)
# ----------------------------
def append_rows_to_csv(rows, out_csv):
    """rows: list of lists matching COLUMNS"""
    if not rows:
        return
    df = pd.DataFrame(rows, columns=COLUMNS)
    header = not os.path.exists(out_csv)
    df.to_csv(out_csv, mode='a', index=False, header=header)
    print(f"[SAVED] {len(df)} rows -> {out_csv}")

# ----------------------------
# Main batch processor (generic)
# ----------------------------
def process_file(input_csv, output_csv, label_value, start=None, end=None, batch_size=BATCH_SIZE):
    print(f"\n=== Processing {input_csv} -> {output_csv} (label={label_value}) ===")
    urls = load_urls_from_csv(input_csv)
    if start is None:
        start = 0
    if end is None:
        end = len(urls)
    urls = urls[start:end]
    print(f"Total URLs to process from file slice: {len(urls)}")

    # Skip already processed URLs (if output exists)
    processed = set()
    if os.path.exists(output_csv):
        try:
            done_df = pd.read_csv(output_csv, usecols=['URL'], dtype=str)
            processed = set(done_df['URL'].astype(str).tolist())
            if processed:
                print(f"Skipping {len(processed)} already-processed URLs from {output_csv}")
        except Exception:
            # If reading fails, continue without skip
            print("[WARN] Could not read existing output to skip processed URLs.")

    # Filter out already processed
    urls = [u for u in urls if u not in processed]
    print(f"Remaining URLs after skip: {len(urls)}")

    # iterate in batches
    for batch_start in range(0, len(urls), batch_size):
        batch = urls[batch_start:batch_start + batch_size]
        rows = []
        print(f"\n--- Batch {batch_start}..{batch_start + len(batch) - 1} ({len(batch)} URLs) ---")
        for i, url in enumerate(batch, start=batch_start + 1):
            soup = scrape_to_soup(url)
            if soup:
                try:
                    vec = fe.create_vector(soup)
                    # ensure vector length matches features count (without URL/label)
                    expected_feat_len = len(COLUMNS) - 2
                    if len(vec) != expected_feat_len:
                        print(f"[WARN] feature vector length mismatch ({len(vec)} != {expected_feat_len}) for URL: {url}")
                        # optionally pad/truncate to expected length:
                        if len(vec) < expected_feat_len:
                            vec = vec + [0] * (expected_feat_len - len(vec))
                        else:
                            vec = vec[:expected_feat_len]
                    vec.append(url)
                    vec.append(label_value)
                    rows.append(vec)
                    print(f"[OK] {i}: {url}")
                except Exception as ex:
                    print(f"[ERROR] feature extraction failed for {url}: {ex}")
            time.sleep(DELAY_BETWEEN_REQUESTS)

        # append this batch to CSV
        append_rows_to_csv(rows, output_csv)

    print(f"[DONE] {input_csv} -> {output_csv}")

# ----------------------------
# Run both datasets
# ----------------------------
if __name__ == "__main__":
    # Legitimate (tranco)
    process_file(
        input_csv=TRUSTED_LEGIT_INPUT,
        output_csv=TRUSTED_LEGIT_OUTPUT,
        label_value=0,
        start=LEGIT_START,
        end=LEGIT_END,
        batch_size=BATCH_SIZE
    )

    # Phishing (verified_online)
    process_file(
        input_csv=PHISH_INPUT,
        output_csv=PHISH_OUTPUT,
        label_value=1,
        start=PHISH_START,
        end=PHISH_END,
        batch_size=BATCH_SIZE
    )

    print("\nAll done.")
# 1 DEFINE A FUNCTION THAT OPENS A HTML FILE AND RETURNS THE CONTENT
file_name = "mini_dataset/9.html"


def open_file(f_name):
    with open(f_name, "r") as f:
        return f.read()


# 2 DEFINE A FUNCTION THAT CREATES A BEATIFULSOUP OBJECT
def create_soup(text):
    return BeautifulSoup(text, "html.parser")


# 3 DEFINE A FUNCTION THAT CREATES A VECTOR BY RUNNING ALL FEATURE FUNCTIONS FOR THE SOUP OBJECT
def create_vector(soup):
    return [
        fe.has_title(soup),
        fe.has_input(soup),
        fe.has_button(soup),
        fe.has_image(soup),
        fe.has_submit(soup),
        fe.has_link(soup),
        fe.has_password(soup),
        fe.has_email_input(soup),
        fe.has_hidden_element(soup),
        fe.has_audio(soup),
        fe.has_video(soup),
        fe.number_of_inputs(soup),
        fe.number_of_buttons(soup),
        fe.number_of_images(soup),
        fe.number_of_option(soup),
        fe.number_of_list(soup),
        fe.number_of_TH(soup),
        fe.number_of_TR(soup),
        fe.number_of_href(soup),
        fe.number_of_paragraph(soup),
        fe.number_of_script(soup),
        fe.length_of_title(soup),
        fe.has_h1(soup),
        fe.has_h2(soup),
        fe.has_h3(soup),
        fe.length_of_text(soup),
        fe.number_of_clickable_button(soup),
        fe.number_of_a(soup),
        fe.number_of_img(soup),
        fe.number_of_div(soup),
        fe.number_of_figure(soup),
        fe.has_footer(soup),
        fe.has_form(soup),
        fe.has_text_area(soup),
        fe.has_iframe(soup),
        fe.has_text_input(soup),
        fe.number_of_meta(soup),
        fe.has_nav(soup),
        fe.has_object(soup),
        fe.has_picture(soup),
        fe.number_of_sources(soup),
        fe.number_of_span(soup),
        fe.number_of_table(soup)
    ]


# 4 RUN STEP 1,2,3 FOR ALL HTML FILES AND CREATE A 2-D ARRAY
folder = "mini_dataset"


def create_2d_list(folder_name):
    directory = os.path.join(os.getcwd(), folder_name)
    data = []
    for file in sorted(os.listdir(directory)):
        soup = create_soup(open_file(directory + "/" + file))
        data.append(create_vector(soup))
    return data

"""
# 5 CREATE A DATAFRAME BY USING 2-D ARRAY
data = create_2d_list(folder)

columns = [
    'has_title',
    'has_input',
    'has_button',
    'has_image',
    'has_submit',
    'has_link',
    'has_password',
    'has_email_input',
    'has_hidden_element',
    'has_audio',
    'has_video',
    'number_of_inputs',
    'number_of_buttons',
    'number_of_images',
    'number_of_option',
    'number_of_list',
    'number_of_th',
    'number_of_tr',
    'number_of_href',
    'number_of_paragraph',
    'number_of_script',
    'length_of_title',
    'has_h1',
    'has_h2',
    'has_h3',
    'length_of_text',
    'number_of_clickable_button',
    'number_of_a',
    'number_of_img',
    'number_of_div',
    'number_of_figure',
    'has_footer',
    'has_form',
    'has_text_area',
    'has_iframe',
    'has_text_input',
    'number_of_meta',
    'has_nav',
    'has_object',
    'has_picture',
    'number_of_sources',
    'number_of_span',
    'number_of_table'
]

df = pd.DataFrame(data=data, columns=columns)

print(df.head(5))
"""