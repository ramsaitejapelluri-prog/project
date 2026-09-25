"""Scrape, clean, normalize, query, and validate the Books to Scrape dataset."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50
OUT = Path(__file__).parent / "output"
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_soup(url: str) -> BeautifulSoup:
    """Fetch a page and return a parsed HTML document."""
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 2:
                continue

    raise RuntimeError(f"Failed to fetch {url} after 3 attempts") from last_error


def scrape_first_three_categories() -> pd.DataFrame:
    """Collect every book from the first three categories, including pagination."""
    homepage = fetch_soup(BASE_URL)
    category_links = homepage.select("div.side_categories ul ul a")[:3]
    rows: list[dict[str, str]] = []

    for link in category_links:
        category = link.get_text(strip=True)
        current_url = urljoin(BASE_URL, str(link["href"]))

        while current_url:
            soup = fetch_soup(current_url)

            for book in soup.select("article.product_pod"):
                title_tag = book.select_one("h3 a")
                price_tag = book.select_one(".price_color")
                rating_tag = book.select_one("p.star-rating")
                availability_tag = book.select_one(".availability")

                if title_tag is None or price_tag is None or rating_tag is None or availability_tag is None:
                    continue

                rows.append(
                    {
                        "title": str(title_tag["title"]),
                        "price_raw": str(price_tag.get_text(strip=True)),
                        "star_rating": str(rating_tag["class"][-1]),
                        "availability_raw": str(availability_tag.get_text(" ", strip=True)),
                        "category": category,
                    }
                )

            next_link = soup.select_one("li.next a")
            current_url = urljoin(current_url, str(next_link["href"])) if next_link else None

    return pd.DataFrame(rows)


def clean_books(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize the raw scraped book data."""
    df = raw.copy()

    df["price_gbp"] = pd.to_numeric(
        df["price_raw"].str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce",
    )
    df["rating"] = df["star_rating"].map(RATING_MAP)
    df["in_stock"] = df["availability_raw"].str.contains("In stock", case=False, na=False)

    df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())
    df["rating"] = df["rating"].fillna(df["rating"].median()).round().astype(int)
    df = df.dropna(subset=["title", "category"])
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    return df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]


def create_database(df: pd.DataFrame, db_path: Path) -> None:
    """Create SQLite tables and store normalized book data."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(
            """
            DROP TABLE IF EXISTS books;
            DROP TABLE IF EXISTS categories;

            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY,
                category_name TEXT NOT NULL UNIQUE
            );

            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                in_stock INTEGER NOT NULL CHECK(in_stock IN (0, 1)),
                category_id INTEGER NOT NULL,
                FOREIGN KEY(category_id) REFERENCES categories(category_id)
            );
            """
        )

        for category in sorted(df["category"].unique()):
            conn.execute("INSERT INTO categories(category_name) VALUES (?)", (category,))

        category_ids = dict(conn.execute("SELECT category_name, category_id FROM categories"))
        category_mapped = df.assign(category_id=df["category"].map(category_ids)).drop(columns="category")
        values = [(*row[:5], int(row[5])) for row in category_mapped.itertuples(index=False, name=None)]

        conn.executemany(
            """
            INSERT INTO books(title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            values,
        )


QUERIES = {
    "in_stock_high_rating": "SELECT title, rating FROM books WHERE in_stock = 1 AND rating >= 4;",
    "most_expensive": "SELECT title, price_gbp FROM books ORDER BY price_gbp DESC LIMIT 10;",
    "distinct_ratings": "SELECT DISTINCT rating FROM books ORDER BY rating;",
    "mid_price_books": "SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 20 AND 40;",
    "selected_categories": "SELECT title FROM books WHERE category_id IN (1, 2);",
    "join_top_rated": """
        SELECT c.category_name, b.title, b.rating, b.price_inr
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE b.rating >= 4
        ORDER BY c.category_name, b.rating DESC
        LIMIT 10;
    """,
}


def execute_queries(db_path: Path, clean: pd.DataFrame) -> None:
    """Run SQL queries, compare them with pandas logic, and save all outputs."""
    with sqlite3.connect(db_path) as conn:
        outputs = {
            name: pd.read_sql(sql, conn).to_dict(orient="records")
            for name, sql in QUERIES.items()
        }

        category_names = clean["category"].drop_duplicates().tolist()
        if category_names:
            placeholders = ", ".join("?" for _ in category_names)
            selected_categories_sql = (
                "SELECT title FROM books WHERE category_id IN ("
                "SELECT category_id FROM categories WHERE category_name IN ({})"
                ")"
            ).format(placeholders)
            selected_categories = pd.read_sql_query(selected_categories_sql, conn, params=category_names)
            outputs["selected_categories"] = selected_categories.to_dict(orient="records")

        join_sql = pd.read_sql(QUERIES["join_top_rated"], conn)
        categories = pd.read_sql("SELECT * FROM categories", conn)

    joined = clean.merge(categories, left_on="category", right_on="category_name")
    join_merge = (
        joined.loc[joined.rating >= 4, ["category_name", "title", "rating", "price_inr"]]
        .sort_values(["category_name", "rating"], ascending=[True, False])
        .head(10)
        .reset_index(drop=True)
    )

    OUT.joinpath("query_outputs.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    OUT.joinpath("queries.sql").write_text(
        "\n\n".join(f"-- {name}\n{sql}" for name, sql in QUERIES.items()),
        encoding="utf-8",
    )
    join_sql.to_csv(OUT / "join_sql.csv", index=False)
    join_merge.to_csv(OUT / "join_pandas_merge.csv", index=False)
    pd.testing.assert_frame_equal(join_sql.reset_index(drop=True), join_merge, check_dtype=False)
    (OUT / "join_validation.txt").write_text("SQL JOIN and pandas merge outputs match exactly.\n", encoding="utf-8")

    print("SQL join result:\n", join_sql)
    print("\nPandas merge result:\n", join_merge)
    print("\nRows scraped:", len(clean))


def main() -> None:
    """Run the full scraping, cleaning, database load, and validation pipeline."""
    OUT.mkdir(exist_ok=True)

    raw = scrape_first_three_categories()
    clean = clean_books(raw)

    if len(clean) < 60 or clean["category"].nunique() < 3:
        raise RuntimeError("Required scope not met: need at least 60 books from 3 categories.")

    clean.to_csv(OUT / "clean_books.csv", index=False)

    db_path = OUT / "books.db"
    create_database(clean, db_path)
    execute_queries(db_path, clean)


if __name__ == "__main__":
    main()
