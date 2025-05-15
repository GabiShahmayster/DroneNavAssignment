import numpy as np

EARTH_RADIUS = 6_378_137  # Meters


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on the Earth's surface using the Haversine formula.

    Args:
    lat1, lon1: Latitude and longitude of the first point (in degrees).
    lat2, lon2: Latitude and longitude of the second point (in degrees).

    Returns:
    Distance in meters.
    """
    # Convert latitude and longitude to radians
    lat1_rad = np.deg2rad(lat1)
    lon1_rad = np.deg2rad(lon1)
    lat2_rad = np.deg2rad(lat2)
    lon2_rad = np.deg2rad(lon2)

    # Calculate differences in coordinates
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    # Haversine formula
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Calculate distance
    distance = EARTH_RADIUS * c
    return distance


def delta_latlon_to_meters(lat1, lon1, lat2, lon2):
    """
    Calculate the differences in meters (dx and dy) between two coordinates.

    Args:
    lat1, lon1: Latitude and longitude of the first point (in degrees).
    lat2, lon2: Latitude and longitude of the second point (in degrees).

    Returns:
    dx and dy differences in meters.

    ^ Y
    |
    |____> X
    """

    dx = haversine_distance(lat1, lon1, lat1, lon2)
    dy = haversine_distance(lat1, lon1, lat2, lon1)

    if lon1 > lon2:
        dx = -dx
    if lat1 > lat2:
        dy = -dy

    return np.array([dx, dy])


def latlon_to_pixels(positions_latlon, mpp, reference_position):
    """
    Convert positions from absolute (latitude, longitude) to:
    1. Delta x and delta y in meters using reference_position
    2. Pixels in the image (i, j)

    :positions: Positions as np.array[(lat, lon), (lat, lon)]
    :mpp: Meters per pixel
    :reference_position: Top left reference coordinate np.array([lat, lon])
    """

    positions = np.array(positions_latlon)

    # Convert positions for meters
    delta_positions_meters = [delta_latlon_to_meters(reference_position[0],
                                                     reference_position[1],
                                                     position[0],
                                                     position[1]
                                                     ) for position in positions]

    # Convert positions to pixels
    delta_positions_meters = np.array(delta_positions_meters)
    delta_positions_pixels = delta_positions_meters / mpp
    delta_positions_pixels = np.round(delta_positions_pixels)

    # Change the direction of Y axis (to fit i axis in pixel space)
    delta_positions_pixels = delta_positions_pixels * np.array([1, -1])
    # Switch x and y to become i and j
    delta_positions_pixels[:, [0, 1]] = delta_positions_pixels[:, [1, 0]]

    return delta_positions_pixels
