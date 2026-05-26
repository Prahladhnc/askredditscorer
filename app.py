import time
import sqlite3
from datetime import datetime, timedelta

import feedparser
import pandas as pd
import pytz
import streamlit as st

import numpy as np
from sentence_transformers import SentenceTransformer, util

# =====================================================
# CONFIG
# =====================================================

RSS_URL = "https://www.reddit.com/r/AskReddit/new/.rss"
DB_NAME = "reddit_posts.db"

REFRESH_SECONDS = 300
POLAND_TZ = pytz.timezone("Europe/Warsaw")

# =====================================================
# STREAMLIT CONFIG
# =====================================================

st.set_page_config(
    page_title="AskReddit Engagement Monitor",
    layout="wide"
)

# =====================================================
# MODEL (TF-BASED EMBEDDINGS)
# =====================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

CATEGORIES = [
    "Relationships",
    "Confessions",
    "Psychology",
    "Social Issues",
    "Ethics",
    "Money",
    "Career",
    "Nostalgia",
    "Controversial",
    "Funny",
    "Hypothetical",
    "Fear",
    "Family",
    "Dating",
    "Technology",
    "Society",
    "Life Advice",
    "Human Behavior",
    "General Discussion"
]

# Precompute category embeddings (IMPORTANT for speed)
category_embeddings = model.encode(CATEGORIES, convert_to_tensor=True)

# =====================================================
# DATABASE
# =====================================================

conn = sqlite3.connect(DB_NAME, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS posts (
    post_id TEXT PRIMARY KEY,
    title TEXT,
    url TEXT,
    posted_time TEXT,
    engagement_score INTEGER,
    reasons TEXT,
    category TEXT,
    fetched_at TEXT
)
""")

conn.commit()

# =====================================================
# TIME CONVERSION
# =====================================================

def convert_to_poland_time(time_str):
    try:
        utc_dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        poland_dt = utc_dt.astimezone(POLAND_TZ)
        return poland_dt.strftime("%d/%m/%Y %H:%M:%S")
    except:
        return time_str

# =====================================================
# DB HELPERS
# =====================================================

def post_exists(post_id):
    cursor.execute("SELECT 1 FROM posts WHERE post_id=?", (post_id,))
    return cursor.fetchone() is not None


def save_post(post_id, title, url, posted_time, score, reason, category):
    cursor.execute("""
        INSERT INTO posts VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        post_id,
        title,
        url,
        posted_time,
        score,
        reason,
        category,
        datetime.utcnow().isoformat()
    ))
    conn.commit()


def cleanup_old_posts():
    cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
    cursor.execute("DELETE FROM posts WHERE fetched_at < ?", (cutoff,))
    conn.commit()

# =====================================================
# SCORING ENGINE (RULE BASED)
# =====================================================

def score_title(title):
    t = title.lower()
    score = 25

    triggers = [
        "would you", "what if", "have you ever",
        "worst", "craziest", "secret", "illegal",
        "regret", "confess", "never told"
    ]
    score += sum(8 for w in triggers if w in t)

    emotions = [
        "hate", "love", "cry", "kill", "cheat",
        "breakup", "divorce", "trauma", "fear"
    ]
    score += sum(6 for w in emotions if w in t)

    if "?" in title:
        score += 10

    words = len(title.split())
    if 8 <= words <= 20:
        score += 10
    elif words > 30:
        score -= 10

    controversial = [
        "politics", "religion", "racist", "drugs",
        "sex", "money", "cheat"
    ]
    score += sum(5 for w in controversial if w in t)

    return max(1, min(100, score))

# =====================================================
# CATEGORY (TF EMBEDDING SIMILARITY)
# =====================================================

def get_category(title):
    emb = model.encode(title, convert_to_tensor=True)
    scores = util.cos_sim(emb, category_embeddings)[0]
    best_idx = int(np.argmax(scores))
    return CATEGORIES[best_idx]

# =====================================================
# ANALYSIS
# =====================================================

def analyze_titles_batch(titles):
    results = []

    for title in titles:
        try:
            category = get_category(title)
            score = score_title(title)

            reasons = []

            if "?" in title:
                reasons.append("question format")

            if any(w in title.lower() for w in ["would you", "what if", "have you ever"]):
                reasons.append("curiosity hook")

            if score > 70:
                reasons.append("high engagement")
            elif score < 40:
                reasons.append("low engagement")

            results.append({
                "title": title,
                "score": score,
                "category": category,
                "reason": ", ".join(reasons) if reasons else "neutral"
            })

        except Exception as e:
            print("Error:", e)
            results.append({
                "title": title,
                "score": 50,
                "category": "General Discussion",
                "reason": "fallback"
            })

    return results

# =====================================================
# FETCH POSTS
# =====================================================

def fetch_posts():
    feed = feedparser.parse(RSS_URL)

    new_entries = []

    for entry in feed.entries:
        try:
            url = entry.link
            post_id = url.split("/comments/")[1].split("/")[0]

            if post_exists(post_id):
                continue

            new_entries.append({
                "post_id": post_id,
                "title": entry.title,
                "url": url,
                "posted_time": entry.published
            })

        except Exception as e:
            print("Parse error:", e)

    BATCH_SIZE = 5
    all_new = []

    for i in range(0, len(new_entries), BATCH_SIZE):
        batch = new_entries[i:i+BATCH_SIZE]
        titles = [b["title"] for b in batch]

        analyses = analyze_titles_batch(titles)

        for item, analysis in zip(batch, analyses):

            save_post(
                item["post_id"],
                item["title"],
                item["url"],
                item["posted_time"],
                analysis["score"],
                analysis["reason"],
                analysis["category"]
            )

            all_new.append({
                "Title": item["title"],
                "Score": analysis["score"],
                "Category": analysis["category"],
                "Posted": convert_to_poland_time(item["posted_time"]),
                "Reason": analysis["reason"],
                "Reddit Link": item["url"]
            })

    return all_new

# =====================================================
# LOAD DATA
# =====================================================

def load_posts(limit=100):
    df = pd.read_sql_query(f"""
        SELECT title, url, posted_time,
               engagement_score, reasons, category, fetched_at
        FROM posts
        ORDER BY datetime(fetched_at) DESC
        LIMIT {limit}
    """, conn)

    df.columns = [
        "Title", "Reddit Link", "Posted",
        "Score", "Reason", "Category", "Fetched At"
    ]

    return df

# =====================================================
# INIT
# =====================================================

cleanup_old_posts()

cursor.execute("SELECT COUNT(*) FROM posts")
if cursor.fetchone()[0] == 0:
    fetch_posts()

# =====================================================
# AUTO REFRESH (STREAMLIT SAFE)
# =====================================================

if "last_run" not in st.session_state:
    st.session_state.last_run = time.time()

if time.time() - st.session_state.last_run > REFRESH_SECONDS:
    st.session_state.last_run = time.time()
    fetch_posts()
    st.rerun()

# =====================================================
# UI
# =====================================================

st.title("🔥 AskReddit Engagement Monitor (TF Version)")

st.markdown("Using TensorFlow embeddings (MiniLM) + rule-based scoring.")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Fetch Latest Posts"):
        fetch_posts()
        st.success("Updated!")

with col2:
    st.info(f"Auto-refresh every {REFRESH_SECONDS // 60} minutes")

df = load_posts()

st.subheader("📋 Latest Posts")
st.dataframe(df, use_container_width=True)

st.subheader("🚀 Top Posts")

if len(df) > 0:
    st.dataframe(
        df.sort_values("Score", ascending=False).head(10),
        use_container_width=True
    )

st.subheader("📊 Stats")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Posts", len(df))

with col2:
    st.metric("Average Score", round(df["Score"].mean(), 1) if len(df) else 0)

with col3:
    st.metric("Max Score", df["Score"].max() if len(df) else 0)

st.caption("RSS Source: https://www.reddit.com/r/AskReddit/new/.rss")