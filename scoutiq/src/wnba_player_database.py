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
    from sportsdataverse.wnba import espn_wnba_team_roster

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
    from sportsdataverse.wnba import espn_wnba_teams_site

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


def list_wnba_teams(db_path=None):
    """Distinct WNBA teams from the local roster SQLite DB.

    Returns abbreviation + display name for Home/Away pickers. Soft-empty when
    the DB is missing so the UI can still fall back to free text.
    """
    path = Path(db_path) if db_path is not None else DB_PATH
    if not path.is_absolute():
        # Resolve relative to scoutiq/ (this file lives in scoutiq/src/).
        path = Path(__file__).resolve().parents[1] / path
    if not path.is_file():
        return []

    conn = sqlite3.connect(path)
    try:
        cursor = conn.cursor()
        rows = cursor.execute(
            """
            SELECT
                team_abbreviation,
                team_display_name,
                team_id,
                COUNT(*) AS player_count
            FROM players
            WHERE team_abbreviation IS NOT NULL
              AND TRIM(team_abbreviation) != ''
            GROUP BY team_abbreviation, team_display_name, team_id
            ORDER BY team_display_name COLLATE NOCASE
            """
        ).fetchall()
    finally:
        conn.close()

    teams = []
    seen = set()
    for abbr, display, team_id, player_count in rows:
        code = str(abbr or "").strip().upper()
        if not code or code in seen:
            continue
        seen.add(code)
        teams.append(
            {
                "abbreviation": code,
                "display_name": str(display or code).strip() or code,
                "team_id": team_id,
                "player_count": int(player_count or 0),
            }
        )
    return teams


if __name__ == "__main__":
    save_all_rosters()
