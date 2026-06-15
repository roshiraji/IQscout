"""
WNBA Player Database Builder

Purpose:
- Retrieve all active WNBA rosters using SportsDataverse
- Store player information in a local SQLite database
- Support future FastAPI endpoints
- Support future RAG integration

Output:
outputs/wnba_players.db
"""

from pathlib import Path
import sqlite3

from sportsdataverse.wnba import espn_wnba_team_roster, espn_wnba_teams_site

DB_PATH = Path("outputs/wnba_players.db")


def create_database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS players (
        athlete_id TEXT PRIMARY KEY,
        athlete_uid TEXT,
        first_name TEXT,
        last_name TEXT,
        full_name TEXT,
        display_name TEXT,
        short_name TEXT,
        jersey TEXT,
        position_id TEXT,
        position_name TEXT,
        position_abbreviation TEXT,
        height REAL,
        display_height TEXT,
        weight REAL,
        display_weight TEXT,
        age INTEGER,
        date_of_birth TEXT,
        birth_city TEXT,
        birth_state TEXT,
        headshot_href TEXT,
        link_web TEXT,
        status_name TEXT,
        team_id INTEGER,
        team_abbreviation TEXT,
        team_display_name TEXT,
        season INTEGER
    )
    """)

    conn.commit()
    return conn


def save_roster(conn, team_id, team_abbreviation=None, team_display_name=None):
    cursor = conn.cursor()
    roster = espn_wnba_team_roster(team_id)

    for player in roster.iter_rows(named=True):
        cursor.execute("""
        INSERT OR REPLACE INTO players VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """, (
            player.get("athlete_id"),
            player.get("athlete_uid"),
            player.get("first_name"),
            player.get("last_name"),
            player.get("full_name"),
            player.get("display_name"),
            player.get("short_name"),
            player.get("jersey"),
            player.get("position_id"),
            player.get("position_name"),
            player.get("position_abbreviation"),
            player.get("height"),
            player.get("display_height"),
            player.get("weight"),
            player.get("display_weight"),
            player.get("age"),
            player.get("date_of_birth"),
            player.get("birth_city"),
            player.get("birth_state"),
            player.get("headshot_href"),
            player.get("link_web"),
            player.get("status_name"),
            player.get("team_id"),
            team_abbreviation,
            team_display_name,
            player.get("season"),
        ))

    conn.commit()
    print(f"Saved {roster.shape[0]} players for {team_display_name} ({team_id})")


def save_all_rosters():
    conn = create_database()
    teams = espn_wnba_teams_site()

    for team in teams.iter_rows(named=True):
        save_roster(
            conn,
            team_id=team["team_id"],
            team_abbreviation=team.get("team_abbreviation"),
            team_display_name=team.get("team_display_name"),
        )

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM players")
    total_players = cursor.fetchone()[0]

    conn.close()

    print()
    print(f"WNBA player database complete.")
    print(f"Total players saved: {total_players}")
    print(f"Database path: {DB_PATH.resolve()}")


if __name__ == "__main__":
    save_all_rosters()