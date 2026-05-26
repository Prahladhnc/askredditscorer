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
REFRESH_SECONDS = 200  # 3 minutes

POLAND_TZ = pytz.timezone("Europe/Warsaw")

# =====================================================
# STREAMLIT
# =====================================================

st.set_page_config(page_title="AskReddit Monitor", layout="wide")

# =====================================================
# MODEL
# =====================================================

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
    reason TEXT,
    category TEXT,
    fetched_at TEXT
)
""")

conn.commit()

# =====================================================
# TIME FORMATTING
# =====================================================

def format_poland_time(ts):
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        dt = dt.astimezone(POLAND_TZ)
        return dt.strftime("%d/%m/%Y %H:%M:%S")
    except:
        return ts

# =====================================================
# DB HELPERS
# =====================================================

def post_exists(post_id):
    cursor.execute("SELECT 1 FROM posts WHERE post_id=?", (post_id,))
    return cursor.fetchone() is not None


def save_post(post_id, title, url, posted_time, score, reason, category):
    cursor.execute("""
        INSERT OR IGNORE INTO posts VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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

# =====================================================
# SCORING
# =====================================================

def score_title(title):
    t = title.lower()
    score = 25

    triggers = [
        "would you","what if","have you ever",
        "worst","craziest","secret","regret"
    ]
    score += sum(8 for w in triggers if w in t)

    emotions = [
        "hate","love","cry","kill","cheat","fear"
    ]
    score += sum(6 for w in emotions if w in t)

    if "?" in title:
        score += 10

    wc = len(title.split())
    if 8 <= wc <= 20:
        score += 10
    elif wc > 30:
        score -= 10

    return max(1, min(100, score))

# =====================================================
# CATEGORY DETECTION
# =====================================================

def get_category(title):
    emb = model.encode(title, convert_to_tensor=True)
    sims = util.cos_sim(emb, category_embeddings)[0]
    return CATEGORIES[int(np.argmax(sims))]

# =====================================================
# ANALYSIS
# =====================================================

def analyze_titles(titles):
    results = []

    for t in titles:
        try:
            cat = get_category(t)
            sc = score_title(t)

            reasons = []
            text = t.lower()

            # 1. Engagement hooks
            hooks = ["would you", "what if", "have you ever", "imagine", "if you could"]
            if any(h in text for h in hooks):
                reasons.append("hypothetical hook")

            # 2. Emotional triggers
            emotions = ["worst", "craziest", "secret", "regret", "trauma", "lost", "cheated"]
            if any(e in text for e in emotions):
                reasons.append("emotional trigger")

            # 3. Controversial / sensitive topics
            controversial = ["racist", "illegal", "drugs", "cheat", "money", "politics", "sex"]
            if any(c in text for c in controversial):
                reasons.append("controversial topic")

            # 4. Length quality signal
            wc = len(t.split())
            if 8 <= wc <= 20:
                reasons.append("ideal length")
            elif wc > 30:
                reasons.append("too long (low retention)")
            elif wc < 6:
                reasons.append("too short (low clarity)")

            # 5. Score-based interpretation
            if sc >= 75:
                reasons.append("high engagement potential")
            elif sc <= 40:
                reasons.append("low engagement potential")

            results.append({
                "score": sc,
                "category": cat,
                "reason": ", ".join(reasons) if reasons else "neutral"
            })

        except:
            results.append({
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

    new_posts = []

    for e in feed.entries:
        try:
            url = e.link
            post_id = url.split("/comments/")[1].split("/")[0]

            if post_exists(post_id):
                continue

            new_posts.append({
                "id": post_id,
                "title": e.title,
                "url": url,
                "time": e.published
            })
        except:
            pass

    BATCH = 5

    for i in range(0, len(new_posts), BATCH):
        batch = new_posts[i:i+BATCH]
        titles = [x["title"] for x in batch]

        results = analyze_titles(titles)

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

def time_ago(seconds):
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    elif seconds < 3600:
        return f"{int(seconds // 60)} minutes ago"
    else:
        return f"{int(seconds // 3600)} hours ago"

# =====================================================
# INIT
# =====================================================

cursor.execute("SELECT COUNT(*) FROM posts")
if cursor.fetchone()[0] == 0:
    fetch_posts()

# =====================================================
# AUTO REFRESH (STREAMLIT SAFE)
# =====================================================

# =====================================================
# AUTO REFRESH (FIXED)
# =====================================================

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

placeholder = st.empty()

# compute time since last refresh
elapsed = time.time() - st.session_state.last_refresh

if elapsed > REFRESH_SECONDS:
    with placeholder:
        st.info("🔄 Auto refreshing feed...")

    fetch_posts()
    st.session_state.last_refresh = time.time()

    st.rerun()
else:
    remaining = REFRESH_SECONDS - int(elapsed)

    st.info(f"⏳ Next auto-refresh in {remaining} seconds")

# =====================================================
# LOAD DATA (ALWAYS FRESH)
# =====================================================

df = pd.read_sql_query("""
SELECT title, url, posted_time,
       engagement_score, reason, category, fetched_at
FROM posts
ORDER BY datetime(fetched_at) DESC
""", conn)

df.columns = [
    "Title","Reddit Link","Posted",
    "Score","Reason","Category","Fetched At"
]

# convert timestamps to Poland time
df["Posted"] = df["Posted"].apply(format_poland_time)
df["Fetched At"] = df["Fetched At"].apply(format_poland_time)

# =====================================================
# UI
# =====================================================

st.title("🔥 AskReddit Engagement Monitor")
now = time.time()
elapsed = now - st.session_state.last_refresh

last_updated = datetime.fromtimestamp(
    st.session_state.last_refresh
).astimezone(POLAND_TZ).strftime("%d/%m/%Y %H:%M:%S")

st.markdown(
    f"""
### ⏱ Feed Status
- 🟢 Last updated: **{last_updated}**
"""
)
st.subheader("📋 Latest Posts (Newest First)")
st.dataframe(df, use_container_width=True)

st.subheader("🚀 Top Posts")
st.dataframe(df.sort_values("Score", ascending=False).head(10),
             use_container_width=True)

# =====================================================
# STATS
# =====================================================

st.subheader("📊 Stats")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Posts", len(df))

with col2:
    st.metric("Average Score",
              round(df["Score"].mean(), 1) if len(df) else 0)

with col3:
    st.metric("Max Score",
              df["Score"].max() if len(df) else 0)

# =====================================================
# REFRESH BUTTON (OPTIONAL MANUAL)
# =====================================================

if st.button("🔄 Refresh Now"):
    fetch_posts()
    st.rerun()

# =====================================================
# FOOTER
# =====================================================

st.caption("Source: Reddit AskReddit RSS Feed")