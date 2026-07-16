from __future__ import annotations

TEAM_ALIASES = {
    "indiana fever": "IND",
    "fever": "IND",
    "ind": "IND",
    "atlanta dream": "ATL",
    "dream": "ATL",
    "atl": "ATL",
    "chicago sky": "CHI",
    "sky": "CHI",
    "chi": "CHI",
    "connecticut sun": "CON",
    "sun": "CON",
    "con": "CON",
    "dallas wings": "DAL",
    "wings": "DAL",
    "dal": "DAL",
    "golden state valkyries": "GS",
    "valkyries": "GS",
    "gs": "GS",
    "las vegas aces": "LV",
    "aces": "LV",
    "lv": "LV",
    "los angeles sparks": "LA",
    "sparks": "LA",
    "la": "LA",
    "minnesota lynx": "MIN",
    "lynx": "MIN",
    "min": "MIN",
    "new york liberty": "NY",
    "liberty": "NY",
    "ny": "NY",
    "phoenix mercury": "PHX",
    "mercury": "PHX",
    "phx": "PHX",
    "portland fire": "POR",
    "fire": "POR",
    "por": "POR",
    "seattle storm": "SEA",
    "storm": "SEA",
    "sea": "SEA",
    "toronto temp": "TOR",
    "temp": "TOR",
    "tor": "TOR",
    "washington mystics": "WSH",
    "mystics": "WSH",
    "wsh": "WSH",
}


def normalize_team_id(value: str | None) -> str | None:
    if not value:
        return None

    cleaned = value.strip().lower()
    return TEAM_ALIASES.get(cleaned, value.strip().upper())


def roster_team_from_jersey_color(
    jersey_color: str | None,
    game_context: dict | None,
) -> str | None:
    """Map broadcast jersey color (red home / blue away) to a roster team_id."""
    if not jersey_color or not game_context:
        return None
    color = jersey_color.strip().lower()
    if color == "red":
        return normalize_team_id(game_context.get("home_team"))
    if color in {"blue", "black", "navy"}:
        return normalize_team_id(game_context.get("away_team"))
    return None
