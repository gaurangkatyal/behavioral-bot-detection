"""
Feature engineering for behavioral fingerprinting of social bots.

Three feature families:
  1. BEHAVIORAL (content-agnostic) — the paper's contribution. These reflect
     account history and operational patterns that an LLM cannot retroactively
     spoof: account-age rates, follow asymmetries, profile completeness,
     screen-name structural properties.
  2. METADATA (semi-content) — profile-text length, has-description, etc.
  3. CONTENT (text-derived) — features of the current tweet text. These are
     the baselines that LLM-laundered text can defeat.
"""

import re
import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone


def parse_twitter_date(s):
    """Parse Twitter created_at format. Returns datetime or NaT."""
    if pd.isna(s) or not isinstance(s, str):
        return pd.NaT
    s = s.strip().strip('"').strip("'")
    for fmt in ("%a %b %d %H:%M:%S %z %Y", "%a %b %d %H:%M:%S +0000 %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return pd.NaT


def char_entropy(s):
    """Shannon entropy of characters in a string (in bits)."""
    if not isinstance(s, str) or not s:
        return 0.0
    counts = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    total = len(s)
    return -sum((n/total) * math.log2(n/total) for n in counts.values())


def digit_ratio(s):
    if not isinstance(s, str) or not s:
        return 0.0
    return sum(c.isdigit() for c in s) / len(s)


def to_bool(v):
    """Coerce string/bool/None to 0/1 robustly."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, np.integer)):
        return int(bool(v))
    if isinstance(v, str):
        s = v.strip().lower()
        return 1 if s in ('true', '1', 'yes', 't') else 0
    return 0


def safe_div(a, b):
    if b is None or b == 0 or pd.isna(b):
        return 0.0
    return a / b


# Reference snapshot: dataset collected ~2017-2018. Use 2018-01-01 as
# reference date for "account age" computation so results are reproducible.
REF_DATE = datetime(2018, 1, 1, tzinfo=timezone.utc)


def engineer_behavioral_features(df, ref_date=REF_DATE):
    """
    Compute content-agnostic behavioral features.
    Input: DataFrame with Twitter user-profile columns.
    Output: DataFrame of engineered features (one row per user).
    """
    out = pd.DataFrame(index=df.index)

    # --- Account-age rates (TRUE behavioral signal: can't be spoofed by text) ---
    created = df['created_at'].apply(parse_twitter_date)
    age_days = ((ref_date - created).dt.total_seconds() / 86400.0).clip(lower=1.0)
    out['account_age_days'] = age_days.fillna(age_days.median())

    out['statuses_per_day'] = df['statuses_count'].astype(float) / out['account_age_days']
    out['followers_per_day'] = df['followers_count'].astype(float) / out['account_age_days']
    out['friends_per_day'] = df['friends_count'].astype(float) / out['account_age_days']
    out['favourites_per_day'] = df['favourites_count'].astype(float) / out['account_age_days']

    # --- Follow asymmetries ---
    out['friends_to_followers'] = df.apply(
        lambda r: safe_div(r['friends_count'], r['followers_count']), axis=1
    )
    out['log_followers'] = np.log1p(df['followers_count'].astype(float))
    out['log_friends'] = np.log1p(df['friends_count'].astype(float))
    out['log_statuses'] = np.log1p(df['statuses_count'].astype(float))
    out['listed_per_follower'] = df.apply(
        lambda r: safe_div(r['listed_count'], r['followers_count']), axis=1
    )

    # --- Profile-completeness signals ---
    out['has_location'] = df['location'].notna().astype(int)
    out['has_description'] = df['description'].notna().astype(int)
    out['has_url'] = df['url'].notna().astype(int)
    out['default_profile'] = df['default_profile'].apply(to_bool)
    out['default_profile_image'] = df['default_profile_image'].apply(to_bool)
    out['verified'] = df['verified'].apply(to_bool)

    # --- Screen-name structural fingerprints ---
    sn = df['screen_name'].fillna('').astype(str).str.strip('"\'')
    out['screen_name_len'] = sn.str.len()
    out['screen_name_digit_ratio'] = sn.apply(digit_ratio)
    out['screen_name_entropy'] = sn.apply(char_entropy)
    out['screen_name_has_trailing_digits'] = sn.apply(
        lambda x: 1 if re.search(r'\d{2,}$', x) else 0
    )

    name = df['name'].fillna('').astype(str).str.strip('"\'')
    out['name_len'] = name.str.len()
    out['name_digit_ratio'] = name.apply(digit_ratio)
    out['name_entropy'] = name.apply(char_entropy)

    # --- Engagement asymmetry ratio ---
    # statuses (outgoing) vs favourites (incoming-flavoured engagement)
    out['statuses_to_favourites'] = df.apply(
        lambda r: safe_div(r['statuses_count'], max(r['favourites_count'], 1)),
        axis=1
    )

    return out


def engineer_content_features(df):
    """
    Content-based features derived from the user's profile description and
    the single sampled tweet. These are the BASELINES that LLM-bots defeat.
    """
    out = pd.DataFrame(index=df.index)

    desc = df['description'].fillna('').astype(str).str.strip('"\'')
    status = df['status'].fillna('').astype(str)

    out['desc_len'] = desc.str.len()
    out['desc_word_count'] = desc.str.split().str.len().fillna(0)
    out['desc_url_count'] = desc.str.count(r'https?://')
    out['desc_hashtag_count'] = desc.str.count(r'#\w+')
    out['desc_mention_count'] = desc.str.count(r'@\w+')
    out['desc_exclaim_count'] = desc.str.count('!')

    out['status_len'] = status.str.len()
    out['status_word_count'] = status.str.split().str.len().fillna(0)
    out['status_url_count'] = status.str.count(r'https?://')
    out['status_hashtag_count'] = status.str.count(r'#\w+')
    out['status_mention_count'] = status.str.count(r'@\w+')
    out['status_uppercase_ratio'] = status.apply(
        lambda x: sum(c.isupper() for c in x) / len(x) if len(x) > 0 else 0.0
    )

    return out


def engineer_all(df):
    """Combine behavioral and content features."""
    b = engineer_behavioral_features(df)
    c = engineer_content_features(df)
    return pd.concat([b, c], axis=1), list(b.columns), list(c.columns)
