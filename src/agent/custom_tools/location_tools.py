"""Location tools using free OpenStreetMap services.

The browser/UI supplies latitude and longitude through the existing graph state.
This module uses:
    - Nominatim for reverse geocoding coordinates into an address.
    - Overpass API for nearby places.

No Google Cloud project, API key, or billing is required.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import requests
from dotenv import load_dotenv
from langchain.tools import tool
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from agent.async_utils import run_in_thread

load_dotenv()

logger = logging.getLogger("location_tools")

NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

DEFAULT_RADIUS_METERS = 3000
HTTP_TIMEOUT_SECONDS = 20
MAX_NEARBY_RESULTS = 10
OSM_USER_AGENT = os.getenv(
    "OSM_USER_AGENT",
    "AndromedaAgent/0.1 (https://github.com/local/andromeda-agent)",
)

PLACE_TYPE_MAP: dict[str, str] = {
    "restaurant": "restaurant",
    "restaurants": "restaurant",
    "hospital": "hospital",
    "hospitals": "hospital",
    "hotel": "hotel",
    "hotels": "hotel",
    "bank": "bank",
    "banks": "bank",
    "atm": "atm",
    "atms": "atm",
    "mosque": "mosque",
    "mosques": "mosque",
    "masjid": "mosque",
    "pharmacy": "pharmacy",
    "pharmacies": "pharmacy",
    "medical store": "pharmacy",
    "petrol pump": "fuel",
    "petrol pumps": "fuel",
    "petrol station": "fuel",
    "petrol stations": "fuel",
    "gas station": "fuel",
    "gas stations": "fuel",
    "fuel station": "fuel",
    "fuel stations": "fuel",
    "shopping mall": "shopping_mall",
    "shopping malls": "shopping_mall",
    "mall": "shopping_mall",
    "malls": "shopping_mall",
    "supermarket": "supermarket",
    "supermarkets": "supermarket",
    "grocery": "supermarket",
    "grocery store": "supermarket",
    "school": "school",
    "schools": "school",
    "park": "park",
    "parks": "park",
    "cafe": "cafe",
    "cafes": "cafe",
    "coffee shop": "cafe",
}

OVERPASS_FILTERS: dict[str, tuple[str, ...]] = {
    "restaurant": ('["amenity"="restaurant"]',),
    "pharmacy": ('["amenity"="pharmacy"]', '["shop"="chemist"]'),
    "hospital": ('["amenity"="hospital"]',),
    "cafe": ('["amenity"="cafe"]',),
    "hotel": ('["tourism"="hotel"]',),
    "bank": ('["amenity"="bank"]',),
    "atm": ('["amenity"="atm"]',),
    "mosque": ('["amenity"="place_of_worship"]["religion"="muslim"]',),
    "fuel": ('["amenity"="fuel"]',),
    "shopping_mall": ('["shop"="mall"]',),
    "supermarket": ('["shop"="supermarket"]',),
    "school": ('["amenity"="school"]',),
    "park": ('["leisure"="park"]',),
}

GENERIC_NEARBY_TYPES = (
    "restaurant",
    "cafe",
    "pharmacy",
    "hospital",
    "bank",
    "atm",
    "fuel",
    "supermarket",
)

LOCATION_KEYWORDS: tuple[str, ...] = (
    "where am i",
    "my location",
    "current location",
    "my current location",
    "live location",
    "show my location",
    "show location",
    "what is my location",
    "what's my location",
    "nearby restaurant",
    "nearby restaurants",
    "nearby hospital",
    "nearby hospitals",
    "nearby hotel",
    "nearby hotels",
    "nearby bank",
    "nearby banks",
    "nearby atm",
    "nearby atms",
    "nearby mosque",
    "nearby mosques",
    "nearby masjid",
    "nearby pharmacy",
    "nearby pharmacies",
    "nearby petrol",
    "nearby gas station",
    "nearby fuel",
    "nearby shopping",
    "nearby mall",
    "nearby supermarket",
    "nearby grocery",
    "nearby school",
    "nearby park",
    "nearby cafe",
    "nearby places",
    "places near me",
    "near me",
    "around me",
    "close to me",
    "what's around me",
    "what is around me",
    "find nearby",
    "search nearby",
    "locate nearby",
)


def _http_session() -> requests.Session:
    """Create a requests session with light retry behavior."""
    retry = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
    )
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": OSM_USER_AGENT,
            "Accept": "application/json",
        }
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Return the distance in kilometres between two geographic points."""
    radius_km = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def reverse_geocode(latitude: float, longitude: float) -> dict[str, Any]:
    """Convert coordinates to a structured address using Nominatim."""
    params = {
        "format": "jsonv2",
        "lat": latitude,
        "lon": longitude,
        "addressdetails": 1,
        "zoom": 18,
    }

    try:
        response = _http_session().get(
            NOMINATIM_REVERSE_URL,
            params=params,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        logger.error("Nominatim reverse geocoding request failed: %s", exc)
        return {
            "formatted_address": "Address unavailable from OpenStreetMap",
            "city": "",
            "state": "",
            "country": "",
            "postal_code": "",
        }

    address = data.get("address", {}) or {}
    city = (
        address.get("city")
        or address.get("town")
        or address.get("village")
        or address.get("municipality")
        or address.get("county")
        or ""
    )

    return {
        "formatted_address": data.get("display_name", ""),
        "city": city,
        "state": address.get("state", ""),
        "country": address.get("country", ""),
        "postal_code": address.get("postcode", ""),
    }


def _overpass_query(
    latitude: float,
    longitude: float,
    place_types: list[str],
    radius: int,
) -> str:
    """Build an Overpass query for the requested place types."""
    selectors: list[str] = []
    for place_type in place_types:
        for tag_filter in OVERPASS_FILTERS.get(place_type, ()):
            selectors.extend(
                [
                    f"node(around:{radius},{latitude},{longitude}){tag_filter};",
                    f"way(around:{radius},{latitude},{longitude}){tag_filter};",
                    f"relation(around:{radius},{latitude},{longitude}){tag_filter};",
                ]
            )

    return "\n".join(
        [
            "[out:json][timeout:20];",
            "(",
            *selectors,
            ");",
            "out center tags;",
        ]
    )


def _element_coordinates(element: dict[str, Any]) -> tuple[float, float] | None:
    """Return coordinates for an Overpass element."""
    if "lat" in element and "lon" in element:
        return float(element["lat"]), float(element["lon"])

    center = element.get("center") or {}
    if "lat" in center and "lon" in center:
        return float(center["lat"]), float(center["lon"])

    return None


def _format_osm_address(tags: dict[str, Any]) -> str:
    """Build a compact address from OSM tags."""
    parts = [
        tags.get("addr:housenumber", ""),
        tags.get("addr:street", ""),
        tags.get("addr:suburb", ""),
        tags.get("addr:city", "") or tags.get("addr:town", ""),
    ]
    return ", ".join(str(part) for part in parts if part)


def _classify_place(tags: dict[str, Any], fallback: str) -> str:
    """Infer a readable place type from OSM tags."""
    if tags.get("amenity") == "place_of_worship" and tags.get("religion") == "muslim":
        return "mosque"
    if tags.get("amenity"):
        return str(tags["amenity"])
    if tags.get("shop"):
        return str(tags["shop"])
    if tags.get("tourism"):
        return str(tags["tourism"])
    if tags.get("leisure"):
        return str(tags["leisure"])
    return fallback


def search_nearby_places(
    latitude: float,
    longitude: float,
    place_type: str = "",
    radius: int = DEFAULT_RADIUS_METERS,
) -> list[dict[str, Any]]:
    """Search nearby places using Overpass/OpenStreetMap."""
    place_types = [place_type] if place_type else list(GENERIC_NEARBY_TYPES)
    query = _overpass_query(latitude, longitude, place_types, radius)

    try:
        response = _http_session().post(
            OVERPASS_URL,
            data={"data": query},
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        logger.error("Overpass nearby search request failed: %s", exc)
        return []

    places: list[dict[str, Any]] = []
    seen: set[tuple[str, float, float]] = set()
    for element in data.get("elements", []):
        coords = _element_coordinates(element)
        if not coords:
            continue

        place_lat, place_lng = coords
        tags = element.get("tags") or {}
        name = tags.get("name") or tags.get("brand") or "Unnamed place"
        detected_type = _classify_place(tags, place_type or "place")
        key = (str(name).lower(), round(place_lat, 5), round(place_lng, 5))
        if key in seen:
            continue
        seen.add(key)

        distance_km = _haversine_distance(latitude, longitude, place_lat, place_lng)
        places.append(
            {
                "name": name,
                "type": detected_type,
                "address": _format_osm_address(tags),
                "distance": f"{distance_km:.2f} km",
                "distance_km": distance_km,
                "latitude": place_lat,
                "longitude": place_lng,
            }
        )

    places.sort(key=lambda place: place["distance_km"])
    return places[:MAX_NEARBY_RESULTS]


def wants_location(text: str) -> bool:
    """Return ``True`` when the user query is location-related."""
    lowered = text.lower()
    return any(keyword in lowered for keyword in LOCATION_KEYWORDS)


def extract_place_types(text: str) -> list[str]:
    """Extract one or more nearby place types from natural-language text."""
    lowered = text.lower()
    place_types: list[str] = []

    for phrase in sorted(PLACE_TYPE_MAP, key=len, reverse=True):
        if phrase in lowered:
            place_type = PLACE_TYPE_MAP[phrase]
            if place_type not in place_types:
                place_types.append(place_type)

    return place_types


def extract_place_type(text: str) -> str:
    """Extract the first matching nearby place type for compatibility."""
    place_types = extract_place_types(text)
    return place_types[0] if place_types else ""


def _wants_nearby_search(user_text: str) -> bool:
    """Return True when the query asks for nearby places."""
    lowered = user_text.lower()
    return any(
        phrase in lowered
        for phrase in ("nearby", "near me", "around me", "close to me", "places near")
    )


def get_location_info_sync(
    latitude: float,
    longitude: float,
    user_text: str = "",
) -> dict[str, Any]:
    """Return structured location info from OSM services."""
    if latitude == 0.0 and longitude == 0.0:
        return {
            "error": True,
            "message": (
                "I need your location to help with this request. "
                "Please allow location access in your browser/device and try again."
            ),
        }

    address_info = reverse_geocode(latitude, longitude)
    result: dict[str, Any] = {
        "current_location": {
            "latitude": latitude,
            "longitude": longitude,
            **address_info,
        },
        "nearby_places": [],
    }

    place_types = extract_place_types(user_text)
    if place_types:
        nearby: list[dict[str, Any]] = []
        for requested_type in place_types:
            nearby.extend(search_nearby_places(latitude, longitude, requested_type))
        nearby.sort(key=lambda place: place.get("distance_km", 999999))
        result["nearby_places"] = nearby[:MAX_NEARBY_RESULTS]
    elif _wants_nearby_search(user_text):
        result["nearby_places"] = search_nearby_places(latitude, longitude, "")

    return result


def _format_location_response(data: dict[str, Any], user_text: str) -> str:
    """Convert structured location data into a natural-language answer."""
    if data.get("error"):
        return str(data["message"])

    location = data.get("current_location", {})
    address = location.get("formatted_address") or "an address I could not resolve"
    city = location.get("city") or "Unknown city"
    state = location.get("state") or "Unknown state/province"
    country = location.get("country") or "Unknown country"
    postal_code = location.get("postal_code") or "not available"

    lines = [
        "Here is what I found from OpenStreetMap:",
        "",
        f"You are at approximately {location.get('latitude')}, {location.get('longitude')}.",
        f"Address: {address}",
        f"City: {city}",
        f"State/Province: {state}",
        f"Country: {country}",
        f"Postal code: {postal_code}",
    ]

    nearby_places = data.get("nearby_places") or []
    if nearby_places:
        requested = ", ".join(
            place_type.replace("_", " ") for place_type in extract_place_types(user_text)
        )
        heading = f"Nearby {requested}" if requested else "Nearby places"
        lines.extend(["", f"{heading}:"])
        for index, place in enumerate(nearby_places, start=1):
            address_part = f" - {place['address']}" if place.get("address") else ""
            lines.append(
                f"{index}. {place['name']} ({place['type']}, {place['distance']})"
                f"{address_part}"
            )
    elif _wants_nearby_search(user_text) or extract_place_types(user_text):
        lines.extend(
            [
                "",
                "I could not find matching nearby places from OpenStreetMap in the search radius.",
            ]
        )

    return "\n".join(lines)


def location_fallback_response(
    user_text: str,
    latitude: float = 0.0,
    longitude: float = 0.0,
) -> str:
    """Return a natural-language location response."""
    data = get_location_info_sync(latitude, longitude, user_text)
    return _format_location_response(data, user_text)


@tool
async def get_live_location(
    latitude: float = 0.0,
    longitude: float = 0.0,
    query: str = "",
) -> str:
    """Get current location details and nearby places using OpenStreetMap.

    Use when the user asks about their location, nearby restaurants, hospitals,
    cafes, ATMs, banks, hotels, mosques, pharmacies, petrol pumps, or nearby
    places. Latitude and longitude should come from browser geolocation.
    """
    return await run_in_thread(location_fallback_response, query, latitude, longitude)


__all__ = [
    "LOCATION_KEYWORDS",
    "extract_place_type",
    "extract_place_types",
    "get_live_location",
    "get_location_info_sync",
    "location_fallback_response",
    "reverse_geocode",
    "search_nearby_places",
    "wants_location",
]
