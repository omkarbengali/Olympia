-- ============================================================
-- OLYMPIA: Olympic Performance Analytics & Intelligence System
-- Star Schema Database Definition for SQLite3
-- ============================================================

PRAGMA foreign_keys = ON;

-- Drop tables if re-initializing (dimension order preserves integrity)
DROP TABLE IF EXISTS fact_medal;
DROP TABLE IF EXISTS dim_event;
DROP TABLE IF EXISTS dim_sport;
DROP TABLE IF EXISTS dim_country;
DROP TABLE IF EXISTS dim_game;
DROP TABLE IF EXISTS dim_medal;

-- ------------------------------------------------------------
-- Dimension 1: dim_game (Olympic Games Editions)
-- ------------------------------------------------------------
CREATE TABLE dim_game (
    game_id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER NOT NULL,
    season TEXT NOT NULL CHECK(season IN ('Summer', 'Winter')),
    games TEXT NOT NULL UNIQUE
);

-- ------------------------------------------------------------
-- Dimension 2: dim_country (NOC and Country Names)
-- ------------------------------------------------------------
CREATE TABLE dim_country (
    country_id INTEGER PRIMARY KEY AUTOINCREMENT,
    country_code TEXT NOT NULL,
    country_name TEXT NOT NULL UNIQUE
);

-- ------------------------------------------------------------
-- Dimension 3: dim_sport (Olympic Disciplines / Sports)
-- ------------------------------------------------------------
CREATE TABLE dim_sport (
    sport_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport_name TEXT NOT NULL UNIQUE
);

-- ------------------------------------------------------------
-- Dimension 4: dim_event (Specific Competitions & Genders)
-- ------------------------------------------------------------
CREATE TABLE dim_event (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport_id INTEGER NOT NULL,
    event_name TEXT NOT NULL,
    event_gender TEXT NOT NULL CHECK(event_gender IN ("Men's", "Women's", "Mixed")),
    FOREIGN KEY (sport_id) REFERENCES dim_sport(sport_id) ON DELETE CASCADE,
    UNIQUE(sport_id, event_name, event_gender)
);

-- ------------------------------------------------------------
-- Dimension 5: dim_medal (Medal Category & Analytical Points)
-- Note: medal_points is an internal analytical weight (Gold=3, Silver=2, Bronze=1)
-- ------------------------------------------------------------
CREATE TABLE dim_medal (
    medal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    medal_name TEXT NOT NULL UNIQUE CHECK(medal_name IN ('Gold', 'Silver', 'Bronze')),
    medal_points INTEGER NOT NULL CHECK(medal_points IN (1, 2, 3))
);

-- ------------------------------------------------------------
-- Fact Table: fact_medal (Individual Medal Awards)
-- ------------------------------------------------------------
CREATE TABLE fact_medal (
    medal_fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    country_id INTEGER NOT NULL,
    sport_id INTEGER NOT NULL,
    event_id INTEGER NOT NULL,
    medal_id INTEGER NOT NULL,
    athletes TEXT,
    FOREIGN KEY (game_id) REFERENCES dim_game(game_id) ON DELETE CASCADE,
    FOREIGN KEY (country_id) REFERENCES dim_country(country_id) ON DELETE CASCADE,
    FOREIGN KEY (sport_id) REFERENCES dim_sport(sport_id) ON DELETE CASCADE,
    FOREIGN KEY (event_id) REFERENCES dim_event(event_id) ON DELETE CASCADE,
    FOREIGN KEY (medal_id) REFERENCES dim_medal(medal_id) ON DELETE CASCADE
);

-- ------------------------------------------------------------
-- Performance Indexes for Fast OLAP Slicing and Joins
-- ------------------------------------------------------------
CREATE INDEX idx_fact_game ON fact_medal(game_id);
CREATE INDEX idx_fact_country ON fact_medal(country_id);
CREATE INDEX idx_fact_sport ON fact_medal(sport_id);
CREATE INDEX idx_fact_event ON fact_medal(event_id);
CREATE INDEX idx_fact_medal ON fact_medal(medal_id);
CREATE INDEX idx_game_year ON dim_game(year);
CREATE INDEX idx_country_code ON dim_country(country_code);
