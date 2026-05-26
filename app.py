import time
import sqlite3
from datetime import datetime, timedelta

import feedparser
import pandas as pd
import pytz
import streamlit as st

import numpy as np
from sentence_transformers import SentenceTransformer, util

# =========================
# CONFIG
# =========================

RSS_URL = "https://www.reddit.com/r/AskReddit/new/.rss"
DB_NAME = "reddit_posts.db"
REFRESH_SECONDS = 300

POLAND_TZ = pytz.timezone("Europe/Warsaw")

# =========================
# STREAMLIT
# =========================

st.set_page_config("AskReddit Monitor", layout="wide")

# =========================
# MODEL (SAFE FOR CLOUD)
# =========================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

CATEGORIES = [
    "Relationships","Confessions","Psychology","Social Issues",
    "Ethics","Money","Career","Nostalgia","Controversial",
    "Funny","Hypothetical","Fear","Family","Dating",
    "Technology","Society","Life Advice","Human Behavior",
    "General Discussion"
]

category_embeddings = model.encode(CATEGORIES, convert_to_tensor=True)

# =========================
# DATABASE
# =========================

conn = sqlite3.connect(DB_NAME, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS posts (
    post_id TEXT PRIMARY KEY,
    title TEXT,
    url TEXT,
    posted_time TEXT,
    engagement_score INTEGER,
    reason TEXT,
    category TEXT,
    fetched_at TEXT
)
""")
conn.commit()

# =========================
# HELPERS
# =========================

def post_exists(post_id):
    cursor.execute("SELECT 1 FROM posts WHERE post_id=?", (post_id,))
    return cursor.fetchone() is not None


def save_post(post_id, title, url, posted_time, score, reason, category):
    cursor.execute("""
        INSERT OR IGNORE INTO posts VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        post_id, title, url, posted_time,
        score, reason, category,
        datetime.utcnow().isoformat()
    ))
    conn.commit()


def convert_time(t):
    try:
        dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
        return dt.astimezone(POLAND_TZ).strftime("%d/%m/%Y %H:%M:%S")
    except:
        return t

# =========================
# SCORING ENGINE
# =========================

def score_title(title):
    t = title.lower()
    score = 25

    triggers = ["would you","what if","have you ever","worst","craziest","secret"]
    score += sum(8 for w in triggers if w in t)

    emotions = ["hate","love","cry","kill","cheat","fear"]
    score += sum(6 for w in emotions if w in t)

    if "?" in title:
        score += 10

    wc = len(title.split())
    if 8 <= wc <= 20:
        score += 10
    elif wc > 30:
        score -= 10

    return max(1, min(100, score))

# =========================
# CATEGORY (EMBEDDINGS)
# =========================

def get_category(title):
    emb = model.encode(title, convert_to_tensor=True)
    sims = util.cos_sim(emb, category_embeddings)[0]
    return CATEGORIES[int(np.argmax(sims))]

# =========================
# ANALYSIS
# =========================

def analyze(titles):
    out = []

    for t in titles:
        try:
            cat = get_category(t)
            sc = score_title(t)

            reason = []
            if "?" in t:
                reason.append("question")
            if sc > 70:
                reason.append("high engagement")

            out.append({
                "score": sc,
                "category": cat,
                "reason": ", ".join(reason) or "neutral"
            })

        except:
            out.append({
                "score": 50,
                "category": "General Discussion",
                "reason": "fallback"
            })

    return out

# =========================
# FETCH
# =========================

def fetch_posts():
    feed = feedparser.parse(RSS_URL)

    new = []

    for e in feed.entries:
        try:
            url = e.link
            post_id = url.split("/comments/")[1].split("/")[0]

            if post_exists(post_id):
                continue

            new.append({
                "id": post_id,
                "title": e.title,
                "url": url,
                "time": e.published
            })
        except:
            pass

    BATCH = 5

    for i in range(0, len(new), BATCH):
        batch = new[i:i+BATCH]
        titles = [x["title"] for x in batch]

        results = analyze(titles)

        for item, res in zip(batch, results):
            save_post(
                item["id"],
                item["title"],
                item["url"],
                item["time"],
                res["score"],
                res["reason"],
                res["category"]
            )

# =========================
# INIT
# =========================

cursor.execute("SELECT COUNT(*) FROM posts")
if cursor.fetchone()[0] == 0:
    fetch_posts()

# =========================
# AUTO REFRESH (SAFE)
# =========================

if "last" not in st.session_state:
    st.session_state.last = time.time()

if time.time() - st.session_state.last > REFRESH_SECONDS:
    st.session_state.last = time.time()
    fetch_posts()
    st.rerun()

# =========================
# UI
# =========================

st.title("🔥 AskReddit Monitor (Cloud Safe)")

df = pd.read_sql_query("""
SELECT title, url, posted_time,
       engagement_score, reason, category, fetched_at
FROM posts
ORDER BY fetched_at DESC
""", conn)

df.columns = ["Title","Link","Posted","Score","Reason","Category","Fetched"]

st.subheader("Posts")
st.dataframe(df, use_container_width=True)

st.subheader("Top 10")
st.dataframe(df.sort_values("Score", ascending=False).head(10))

st.metric("Total Posts", len(df))
st.metric("Avg Score", round(df["Score"].mean(),1) if len(df) else 0)