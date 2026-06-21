"""
app.py
CineMatch-ML — Real-time Personalized Movie Recommendation System
Run with: streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import joblib
import sys
import os
import random
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.data_loader import load_movielens
from src.hybrid.hybrid_recommender import HybridRecommender

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch-ML",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme State ────────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

dark = st.session_state.dark_mode

if dark:
    bg       = "#0a0a0f"
    sidebar  = "#111118"
    card_bg  = "#1a1a24"
    card_bdr = "#2a2a3a"
    text     = "#e8e8e8"
    subtext  = "#888"
    accent   = "#e50914"
    accent2  = "#ffd700"
    divider  = "#222"
else:
    bg       = "#f5f5f0"
    sidebar  = "#ffffff"
    card_bg  = "#ffffff"
    card_bdr = "#e0e0e0"
    text     = "#1a1a2e"
    subtext  = "#666"
    accent   = "#e50914"
    accent2  = "#d4900a"
    divider  = "#e0e0e0"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&display=swap');

[data-testid="stAppViewContainer"] {{ background: {bg} !important; color: {text}; }}
[data-testid="stSidebar"] {{ background: {sidebar} !important; border-right: 1px solid {divider}; }}
[data-testid="stSidebar"] * {{ color: {text} !important; }}

.hero-title {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem; letter-spacing: 4px;
    background: linear-gradient(135deg, {accent}, {accent2});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1;
}}
.hero-sub {{ font-size: 0.85rem; color: {subtext}; letter-spacing: 3px; text-transform: uppercase; margin-top: 6px; }}
.section-header {{
    font-family: 'Bebas Neue', sans-serif; font-size: 1.3rem;
    letter-spacing: 3px; color: {text};
    border-bottom: 2px solid {accent}; padding-bottom: 6px; margin-bottom: 16px;
}}
.metric-card {{
    background: {card_bg}; border: 1px solid {card_bdr};
    border-radius: 12px; padding: 18px; text-align: center;
    transition: border-color 0.3s, transform 0.2s;
}}
.metric-card:hover {{ border-color: {accent}; transform: translateY(-2px); }}
.metric-value {{ font-family: 'Bebas Neue', sans-serif; font-size: 2rem; color: {accent}; line-height: 1; }}
.metric-label {{ font-size: 0.72rem; color: {subtext}; text-transform: uppercase; letter-spacing: 2px; margin-top: 4px; }}
.movie-card {{
    background: {card_bg}; border: 1px solid {card_bdr};
    border-radius: 12px; padding: 14px 18px; margin-bottom: 8px;
    display: flex; align-items: center; gap: 14px; transition: all 0.25s;
}}
.movie-card:hover {{ border-color: {accent}; transform: translateX(5px); box-shadow: 0 4px 20px rgba(229,9,20,0.15); }}
.movie-rank {{ font-family: 'Bebas Neue', sans-serif; font-size: 1.8rem; color: {card_bdr}; min-width: 38px; }}
.movie-title {{ font-size: 0.9rem; font-weight: 500; color: {text}; flex: 1; }}
.score-pill {{ display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; color: white; }}
.trend-card {{ background: {card_bg}; border: 1px solid {card_bdr}; border-radius: 12px; padding: 16px; margin-bottom: 8px; }}

div[data-testid="stButton"] > button {{
    background: linear-gradient(135deg, {accent}, #c40812) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; padding: 10px 24px !important;
    font-weight: 600 !important; width: 100% !important; letter-spacing: 1px !important;
}}
div[data-testid="stButton"] > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(229,9,20,0.4) !important;
}}
[data-testid="stSidebar"] div[data-testid="stButton"] > button {{
    width: auto !important;
    padding: 3px 10px !important;
    font-size: 0.72rem !important;
    letter-spacing: 0px !important;
    background: {card_bg} !important;
    border: 1px solid {card_bdr} !important;
    color: {text} !important;
}}

/* ── Deploy button: visible, theme-aware ── */
[data-testid="stToolbar"] {{
    background: {card_bg} !important;
    border-bottom: 1px solid {card_bdr} !important;
}}
[data-testid="stToolbar"] * {{
    color: {text} !important;
    fill: {text} !important;
}}
[data-testid="stToolbar"] button {{
    background: {card_bg} !important;
    border: 1px solid {card_bdr} !important;
    border-radius: 6px !important;
    color: {text} !important;
}}
[data-testid="stToolbar"] button:hover {{
    border-color: {accent} !important;
    box-shadow: 0 4px 15px rgba(229,9,20,0.3) !important;
    background: {bg} !important;
}}

/* ── Download Recommendations button: theme-aware ── */
[data-testid="stDownloadButton"] > button {{
    background: {bg} !important;
    color: {text} !important;
    border: 1px solid {card_bdr} !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    width: 100% !important;
    letter-spacing: 1px !important;
}}
[data-testid="stDownloadButton"] > button:hover {{
    border-color: {accent} !important;
    box-shadow: 0 4px 15px rgba(229,9,20,0.4) !important;
    transform: translateY(-1px) !important;
}}

#MainMenu {{visibility: hidden;}} footer {{visibility: hidden;}}
hr {{ border-color: {divider} !important; }}
</style>
""", unsafe_allow_html=True)


# ── Load Resources ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    ratings_df, movies_df = load_movielens(
        ratings_path="data/raw/u.data",
        movies_path="data/raw/u.item",
    )
    cf_model      = joblib.load("models/svd_model.pkl")
    content_model = joblib.load("models/content_model.pkl")
    hybrid        = HybridRecommender(cf_model=cf_model, content_model=content_model, alpha=0.7)
    return ratings_df, movies_df, hybrid

try:
    ratings_df, movies_df, hybrid = load_resources()
    models_loaded = True
except Exception as e:
    models_loaded = False
    error_msg = str(e)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding:16px 0;'>
        <div style='font-family:Bebas Neue,sans-serif; font-size:1.8rem;
            background:linear-gradient(135deg,{accent},{accent2});
            -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:3px;'>
            CINEMATCH
        </div>
        <div style='font-size:0.6rem; color:{subtext}; letter-spacing:3px;'>ML RECOMMENDATION ENGINE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Theme toggle
    if st.button("🌙 Dark" if dark else "☀️ Light", key="theme_toggle"):
        st.session_state.dark_mode = not dark
        st.rerun()

    st.markdown("---")
    st.markdown(f"<div style='font-size:0.7rem; color:{subtext}; text-transform:uppercase; letter-spacing:2px; margin-bottom:8px;'>⚙️ Settings</div>", unsafe_allow_html=True)
    n_recommendations = st.slider("Recommendations", 5, 20, 10)
    alpha = st.slider("CF Weight (Alpha)", 0.0, 1.0, 0.7, 0.1)

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.78rem; color:{subtext}; line-height:2;'>
        🤝 CF: <b style='color:{accent}'>{alpha:.1f}</b> &nbsp;
        🎭 Content: <b style='color:{accent2}'>{round(1-alpha,1)}</b>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<div style='font-size:0.7rem; color:{subtext}; text-transform:uppercase; letter-spacing:2px; margin-bottom:8px;'>📍 Pages</div>", unsafe_allow_html=True)
    page = st.radio("Navigation", ["🎯 Recommendations", "📊 Analytics", "🔥 Trending", "ℹ️ About"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(f"<div style='font-size:0.65rem; color:{subtext}; text-align:center;'>MovieLens 100K · SVD + TF-IDF<br>Built by MEGHANA_KAMATAM</div>", unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='padding:24px 0 16px 0;'>
    <div class='hero-title'>🎬 CineMatch-ML</div>
    <div class='hero-sub'>Real-time Personalized Movie Recommendations · MovieLens 100K</div>
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error(f"⚠️ Models not loaded: {error_msg}")
    st.stop()

# ── Metrics ────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, val, label in zip(
    [c1, c2, c3, c4],
    [ratings_df['user_id'].nunique(), ratings_df['item_id'].nunique(), len(ratings_df), f"{ratings_df['rating'].mean():.2f}"],
    ["👥 Users", "🎬 Movies", "⭐ Ratings", "📈 Avg Rating"]
):
    col.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{val}</div>
        <div class='metric-label'>{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE 1 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════
if page == "🎯 Recommendations":
    left, right = st.columns([1, 2])

    with left:
        st.markdown('<div class="section-header">🎯 GET RECOMMENDATIONS</div>', unsafe_allow_html=True)
        user_id = st.number_input("Enter User ID (1–943)", min_value=1, max_value=943, value=1)

        if st.button("🎲 Random User"):
            st.session_state["rand_user"] = random.randint(1, 943)
            st.rerun()
        if "rand_user" in st.session_state:
            user_id = st.session_state["rand_user"]
            st.markdown(f"<div style='color:{accent}; font-size:0.8rem;'>🎲 Random User: #{user_id}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 FIND MY MOVIES"):
            with st.spinner("🎬 Analysing your taste..."):
                hybrid.alpha = alpha
                recs = hybrid.recommend(user_id, movies_df, n=n_recommendations)
            st.session_state["recommendations"] = recs
            st.session_state["user_id"] = user_id

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🤖 MODEL INFO</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style='font-size:0.8rem; color:{subtext}; line-height:2.2;'>
            📉 SVD RMSE: <b style='color:{accent}'>~0.93</b><br>
            📉 Hybrid RMSE: <b style='color:{accent2}'>~0.91</b><br>
            ✅ Tests: <b style='color:#4caf50'>23/23 Passing</b>
        </div>""", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-header">🎥 TOP RECOMMENDATIONS</div>', unsafe_allow_html=True)

        if "recommendations" in st.session_state:
            recs = st.session_state["recommendations"]
            uid  = st.session_state["user_id"]

            st.markdown(f"""<div style='font-size:0.78rem; color:{subtext}; margin-bottom:14px;'>
                TOP {len(recs)} PICKS FOR USER <span style='color:{accent}; font-weight:600'>#{uid}</span>
                &nbsp;·&nbsp; ALPHA <span style='color:{accent2}'>{alpha}</span>
            </div>""", unsafe_allow_html=True)

            for i, row in recs.iterrows():
                score_pct = int(row["hybrid_score"] * 100)
                color = "#4caf50" if score_pct >= 70 else "#ffd700" if score_pct >= 50 else accent
                st.markdown(f"""
                <div class='movie-card'>
                    <div class='movie-rank'>#{i+1:02d}</div>
                    <div class='movie-title'>{row['title']}</div>
                    <span class='score-pill' style='background:{color};'>{score_pct}% match</span>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            csv = recs[["title", "hybrid_score"]].to_csv(index=False)
            st.download_button("⬇️ Download Recommendations", csv,
                               file_name=f"recs_user_{uid}.csv", mime="text/csv")
        else:
            st.markdown(f"""
            <div style='background:{card_bg}; border:1px dashed {card_bdr};
                border-radius:12px; padding:60px; text-align:center; color:{subtext};'>
                <div style='font-size:3rem; margin-bottom:12px;'>🎬</div>
                <div style='font-family:Bebas Neue,sans-serif; font-size:1.2rem; letter-spacing:2px;'>YOUR MOVIES AWAIT</div>
                <div style='font-size:0.8rem; margin-top:8px;'>Enter a User ID and click Find My Movies</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE 2 — ANALYTICS
# ══════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown('<div class="section-header">📊 DATASET ANALYTICS</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor(card_bg)
        ax.set_facecolor(card_bg)
        ratings_df['rating'].value_counts().sort_index().plot(kind='bar', ax=ax, color=accent, edgecolor='none')
        ax.tick_params(colors=text); ax.spines[:].set_color(card_bdr)
        ax.set_title('Rating Distribution', color=text, fontsize=11)
        ax.set_xlabel('Rating', color=subtext); ax.set_ylabel('Count', color=subtext)
        plt.xticks(rotation=0); plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        top10 = ratings_df.groupby('item_id').size().nlargest(10).reset_index()
        top10.columns = ['item_id', 'count']
        top10 = top10.merge(movies_df[['item_id', 'title']], on='item_id')
        top10['title'] = top10['title'].str[:22] + '...'
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor(card_bg); ax.set_facecolor(card_bg)
        ax.barh(top10['title'], top10['count'], color=accent2, edgecolor='none')
        ax.tick_params(colors=text, labelsize=7); ax.spines[:].set_color(card_bdr)
        ax.invert_yaxis(); ax.set_title('Top 10 Most Rated', color=text, fontsize=11)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    genre_cols = ['Action','Adventure','Animation','Childrens','Comedy','Crime',
                  'Documentary','Drama','Fantasy','Film_Noir','Horror','Musical',
                  'Mystery','Romance','Sci_Fi','Thriller','War','Western']
    genre_counts = movies_df[genre_cols].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_facecolor(card_bg); ax.set_facecolor(card_bg)
    bar_colors = [accent if i == 0 else accent2 if i == 1 else card_bdr for i in range(len(genre_counts))]
    ax.bar(genre_counts.index, genre_counts.values, color=bar_colors, edgecolor='none')
    ax.tick_params(colors=text, labelsize=8); ax.spines[:].set_color(card_bdr)
    plt.xticks(rotation=45, ha='right'); ax.set_title('Genre Distribution', color=text, fontsize=11)
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════
# PAGE 3 — TRENDING
# ══════════════════════════════════════════════════════
elif page == "🔥 Trending":
    st.markdown('<div class="section-header">🔥 TRENDING MOVIES</div>', unsafe_allow_html=True)

    top_movies = ratings_df.groupby('item_id').agg(
        count=('rating','count'), avg_rating=('rating','mean')).reset_index()
    top_movies = top_movies[top_movies['count'] >= 50]
    top_movies = top_movies.merge(movies_df[['item_id','title']], on='item_id')
    top_movies['score'] = top_movies['avg_rating'] * top_movies['count'].apply(lambda x: min(x/500,1))
    top_movies = top_movies.sort_values('score', ascending=False).head(20)

    col1, col2 = st.columns(2)
    for i, (_, row) in enumerate(top_movies.iterrows()):
        col = col1 if i % 2 == 0 else col2
        stars = "⭐" * int(round(row['avg_rating']))
        col.markdown(f"""
        <div class='trend-card'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <div style='font-size:0.85rem; font-weight:500; color:{text};'>#{i+1} {row['title']}</div>
                    <div style='font-size:0.72rem; color:{subtext}; margin-top:4px;'>
                        {stars} {row['avg_rating']:.1f}/5 · {int(row['count'])} ratings
                    </div>
                </div>
                <div style='font-family:Bebas Neue,sans-serif; font-size:1.4rem; color:{accent};'>
                    {int(row['score']*100)}
                </div>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ══════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown('<div class="section-header">ℹ️ ABOUT CINEMATCH-ML</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='trend-card'>
            <div style='font-family:Bebas Neue,sans-serif; font-size:1.1rem; letter-spacing:2px; color:{text}; margin-bottom:10px;'>🤖 MODELS</div>
            <div style='font-size:0.85rem; color:{subtext}; line-height:2.2;'>
                ✅ <b style='color:{text}'>SVD</b> — Matrix Factorization<br>
                ✅ <b style='color:{text}'>NMF</b> — Non-negative MF<br>
                ✅ <b style='color:{text}'>User-CF</b> — User Collaborative<br>
                ✅ <b style='color:{text}'>Item-CF</b> — Item Collaborative<br>
                ✅ <b style='color:{text}'>TF-IDF</b> — Content-Based<br>
                🏆 <b style='color:{accent}'>Hybrid</b> — SVD + TF-IDF
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='trend-card'>
            <div style='font-family:Bebas Neue,sans-serif; font-size:1.1rem; letter-spacing:2px; color:{text}; margin-bottom:10px;'>📊 PERFORMANCE</div>
            <div style='font-size:0.85rem; color:{subtext}; line-height:2.2;'>
                User-CF RMSE: <b style='color:{text}'>~1.02</b><br>
                Item-CF RMSE: <b style='color:{text}'>~0.98</b><br>
                NMF RMSE: <b style='color:{text}'>~0.96</b><br>
                SVD RMSE: <b style='color:{accent2}'>~0.93</b><br>
                🏆 Hybrid RMSE: <b style='color:{accent}'>~0.91</b><br>
                ✅ Tests: <b style='color:#4caf50'>23/23 Passing</b>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='trend-card' style='text-align:center; margin-top:12px;'>
        <div style='font-size:0.9rem; color:{subtext};'>
            Built by <b style='color:{accent}'>MEGHANARA123456</b> · Data Science & ML Intern<br>
            MovieLens 100K · Python · Scikit-learn · Scikit-surprise · Streamlit
        </div>
    </div>""", unsafe_allow_html=True)