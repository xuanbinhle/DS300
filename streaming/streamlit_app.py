# app.py
# -*- coding: utf-8 -*-

import streamlit as st
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
import os
from pathlib import Path
from fastapi_sending import send_to_kafka_via_fastapi
from fastapi_sending import get_to_kafka_via_fastapi
import time


FASTAPI_PRODUCER_URL = "http://localhost:8000/produce-message"
FASTAPI_CONSUMER_URL = 'http://localhost:8001/drain?limit=10'

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_path = os.path.join(project_root, 'data', 'final', 'final_cleaned_books.csv')
book_df = pd.read_csv(data_path)

st.set_page_config(page_title="Book Recommender Demo", page_icon="📚", layout="wide")

# -----------------------------
# Data contracts (replace later)
# -----------------------------
@dataclass
class Book:
    id: str
    title: str
    author: str
    genre: str
    description: str
    cover_url: str
    product_index: int
    rating: float = 4.5
    price: str = "$9.99"
# Demo data (replace by your DB)
BOOKS: List[Book] = []

for book in book_df.iterrows():
    BOOKS.append(
        Book(
            id=book[1]['product_id'],
            title=book[1]['product_name'],
            author=book[1]['authors'],
            genre=book[1]['type_book'],
            description=book[1]['description'],
            cover_url=book[1]['image'],
            rating=float(book[1]['rating_average']),
            price=f"${book[1]['price']}",
            product_index=int(book[1]['product_index'])
        )
    )

def get_book(book_id: str) -> Book:
    return next(b for b in BOOKS if b.id == book_id)

def get_recommendations(FASTAPI_URL: str, k: int = 10) -> List[Book]:
    # Không cần sleep ở đây vì get_to_kafka_via_fastapi đã tự động đợi
    list_product_index = get_to_kafka_via_fastapi(FASTAPI_URL, max_wait_time=60)
    
    if not list_product_index:
        st.warning("⚠ Không nhận được recommendations, hiển thị sách ngẫu nhiên")
        return BOOKS[:k]
    
    base = [b for b in BOOKS if b.product_index in list_product_index]
    return (base * 3)[:k]

# -----------------------------
# UI helpers
# -----------------------------
def inject_css():
    # Load Font Awesome
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <style>
        .feature-grid{
        display: grid;
        grid-template-columns: repeat(5, 210px); /* set cứng width mỗi card */
        gap: 18px;
        align-items: stretch;
        overflow-x: auto;            /* nhỏ quá thì scroll ngang */
        padding-bottom: 8px;
        }

        /* Card height cố định để không nhảy */
        .book-card{
        width: 210px;
        height: 420px;               /* set cứng chiều cao card */
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.02);
        padding: 12px;
        box-sizing: border-box;

        display: flex;               /* để đẩy nút xuống đáy */
        flex-direction: column;
        }

        /* Cover cố định tỉ lệ + không kéo dãn */
        .book-cover{
        width: 100%;
        height: 180px;               /* set cứng chiều cao ảnh */
        border-radius: 12px;
        overflow: hidden;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        }

        .book-cover img{
            width: 100%;
            height: 100%;
            object-fit: cover;           /* không méo ảnh */
            display: block;
        }

        /* Title/genre/rating: clamp để không làm card cao thấp khác nhau */
        .book-title{
            display: block;
            white-space: nowrap;
            overflow: hidden;  /* Prevents overflow */
            text-overflow: ellipsis;  /* Adds '...' for overflow text */
            font-weight: bold;
        }
        .book-meta{
            margin-top: 8px;
            font-size: 12px;
            opacity: 0.85;
            min-height: 34px;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .book-rating{
            margin-top: 8px;
            min-height: 22px;
        }
        .book-price{
            margin-top: 10px;
            font-weight: 800;
        }

        /* Spacer đẩy button xuống đáy */
        .spacer{ flex: 1; }

        /* Nút full width, không lệch */
        .book-btn button{
            width: 100% !important;
        }

        .book-desc {
            margin-top: 6px;
            font-size: 14px;
            line-height: 1.5; 
        }

        .desc-clamp {
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        
        /* Star rating styles */
        .fa-star, .fa-star-half-alt {
            color: #FFD700;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def rating_star(rating):
    # Font Awesome classes for stars
    full_star = '<i class="fas fa-star"></i>'  # Solid full star
    half_star = '<i class="fas fa-star-half-alt"></i>'  # Half-filled star
    empty_star = '<i class="far fa-star"></i>'  # Empty star

    full_stars = int(rating)  # Full stars (integer part)
    half_stars = 1 if rating % 1 >= 0.5 else 0  # Half stars (if there is a decimal part >= 0.5)
    empty_stars = 5 - full_stars - half_stars  # Remaining stars are empty

    star_string = (full_star * full_stars) + (half_star * half_stars) + (empty_star * empty_stars)
    
    return star_string

def book_tile(book: Book, key_prefix: str = "", show_button: bool = True):
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(book.cover_url, use_container_width=True)
        if len((book.title).split()) > 2:
            title = ' '.join((book.title).split()[:2]) + "..."
            st.markdown(f"**{title}**")
        else:
            st.markdown(f"**{book.title}**")
        st.markdown(f"<span class='pill'>{rating_star(book.rating)} {book.rating:.1f}</span>", unsafe_allow_html=True)
        st.markdown(f"<span class='price'>{(book.price).replace('$','')}₫</span>", unsafe_allow_html=True)
        
        if show_button:
            if st.button("View", key=f"{key_prefix}view_{book.id}", use_container_width=True):
                st.session_state["selected_book_id"] = book.id
                try:
                    send_to_kafka_via_fastapi(book.product_index, FASTAPI_PRODUCER_URL)
                    # st.success(resp.get("status", "Sent"))
                except Exception as e:
                    st.error(f"Failed to send: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Session state
# -----------------------------
if "selected_book_id" not in st.session_state:
    st.session_state["selected_book_id"] = None

inject_css()

# -----------------------------
# Header / Search (commerce feBookel)
# -----------------------------
left, right = st.columns([3, 2])
with left:
    st.markdown("<div class='app-title'>📚 BookStore Recommender</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Click a book cover to see details and personalized similar books.</div>", unsafe_allow_html=True)

with right:
    q = st.text_input("Search", placeholder="Search title / author / genre...")
    genre_filter = st.selectbox("Genre", ["All"] + sorted({b.genre for b in BOOKS}))
    st.caption("Demo UX/UI (commerce-style grid → detail → recommendations)")

# Filtered listing
filtered = BOOKS
if q.strip():
    qq = q.lower().strip()
    filtered = [b for b in filtered if qq in b.title.lower() or qq in b.author.lower() or qq in b.genre.lower()]
if genre_filter != "All":
    filtered = [b for b in filtered if b.genre == genre_filter]

# -----------------------------
# Main layout: grid + detail
# -----------------------------
grid_col, detail_col = st.columns([1, 1], gap="large")

with grid_col:
    st.subheader("Featured Books")
    cols = st.columns(5, gap="medium")
    for i, b in enumerate(filtered[:10]):  # show up to 10
        with cols[i % 5]:
            book_tile(b, key_prefix="grid_")

    if not filtered:
        st.info("No books match your search.")

with detail_col:
    st.subheader(" Details")

    if st.session_state["selected_book_id"] is None:
        st.markdown("<div class='card'>Select a book from the left grid to view details.</div>", unsafe_allow_html=True)
    else:
        book = get_book(st.session_state["selected_book_id"])

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        recs = get_recommendations(k=10, FASTAPI_URL=FASTAPI_CONSUMER_URL)

        if recs:

            c1, c2 = st.columns([1, 1.4], gap="medium")
            with c1:
                st.image(book.cover_url, use_container_width=True)
            with c2:
                st.markdown(f"**{book.title}**")
                st.markdown(f"**Author:** {book.author}")
                st.markdown(f"**Genre:** {book.genre}")
                st.markdown(f"<span class='pill'>{rating_star(book.rating)} {book.rating:.1f}</span>", unsafe_allow_html=True)
                st.markdown(f"<span class='price'>{(book.price).replace('$','')}₫</span>", unsafe_allow_html=True)
                st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
                st.markdown("**Description**")
                desc_key = f"desc_expand_{book.id}"
                if desc_key not in st.session_state:
                    st.session_state[desc_key] = False
                expanded = st.session_state[desc_key]
                desc_class = "book-desc" if expanded else "book-desc desc-clamp"

                st.markdown(
                    f"""
                    <div class="{desc_class}">
                        {book.description}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # View more / less button
                label = "View less ▲" if expanded else "View more ▼"
                if st.button(label, key=f"toggle_desc_{book.id}"):
                    st.session_state[desc_key] = not expanded

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("")
            st.subheader("Recommended Similar Books")        

            # display recs as horizontal-like grid (2 rows x 5)
            rec_cols = st.columns(5, gap="medium")
            for i, rb in enumerate(recs[:10]):
                with rec_cols[i % 5]:
                    book_tile(rb, key_prefix="rec_", show_button=False)
