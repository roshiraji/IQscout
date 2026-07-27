"""WNBA roster team listing for Home/Away pickers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.wnba_player_database import list_wnba_teams


def test_list_wnba_teams_from_db(tmp_path: Path):
    db = tmp_path / "wnba_players.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE players (
            athlete_id TEXT PRIMARY KEY,
            team_id INTEGER,
            team_abbreviation TEXT,
            team_display_name TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO players VALUES (?, ?, ?, ?)",
        [
            ("1", 5, "IND", "Indiana Fever"),
            ("2", 5, "IND", "Indiana Fever"),
            ("3", 20, "ATL", "Atlanta Dream"),
            ("4", 20, None, None),
        ],
    )
    conn.commit()
    conn.close()

    teams = list_wnba_teams(db)
    assert [t["abbreviation"] for t in teams] == ["ATL", "IND"]
    assert teams[1]["display_name"] == "Indiana Fever"
    assert teams[1]["player_count"] == 2


def test_list_wnba_teams_missing_db(tmp_path: Path):
    assert list_wnba_teams(tmp_path / "missing.db") == []
