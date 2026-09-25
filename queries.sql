-- in_stock_high_rating
SELECT title, rating FROM books WHERE in_stock = 1 AND rating >= 4;

-- most_expensive
SELECT title, price_gbp FROM books ORDER BY price_gbp DESC LIMIT 10;

-- distinct_ratings
SELECT DISTINCT rating FROM books ORDER BY rating;

-- mid_price_books
SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 20 AND 40;

-- selected_categories
SELECT title FROM books WHERE category_id IN (1, 2);

-- join_top_rated

        SELECT c.category_name, b.title, b.rating, b.price_inr
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE b.rating >= 4
        ORDER BY c.category_name, b.rating DESC
        LIMIT 10;
    