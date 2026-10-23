import math

import numpy as np

C = 299792458  # speed of light in meters per second


def distance(
    A: np.ndarray, B: np.ndarray
) -> float:
    """Calculates the Euclidean distance between points A and B."""
    return math.sqrt((B[0] - A[0]) ** 2 + (B[1] - A[1]) ** 2 + (B[2] - A[2]) ** 2)


def heading_vector(
    A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """Calculates the a normalised vector pointing from point A to point B."""
    heading_vector = (np.array(B) - np.array(A)) / distance(A, B)
    return heading_vector


def fresnel_zone_radius(
    A: np.ndarray, B: np.ndarray, n: int, frequency_hz: float, d1_m: float
) -> float:
    """Calculates the radius of the nth Fresnel zone at a given distance from point A"""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise TypeError("n must be an integer.")
    
    if n < 1:
        raise ValueError("n must be a positive integer.")
        
    if frequency_hz <= 0:
        raise ValueError("Frequency must be a positive value.")
    
    D_m = distance(A, B)
    
    if D_m <= 0:
        raise ValueError("Distance between points A and B must be greater than zero.")
    
    if D_m <= d1_m:
        raise ValueError(
            f"Distance from point A to the point of interest must be"
            f"less than the distance between points A and B ({D_m} m)."
        )
    
    d2_m = D_m - d1_m
    wavelength_m = C / frequency_hz
    radius_m = math.sqrt(n * wavelength_m * d1_m * d2_m / D_m)
    
    return radius_m


def theoretical_volume(
    A: np.ndarray, B: np.ndarray, frequency_hz: float
) -> float:
    """Calculates the volume of the ellipsoid defined by points A and B."""
    
    semi_major_axis_length_m = distance(A, B) / 2
    
    if semi_major_axis_length_m <= 0:
        raise ValueError("Semi-major axis length must be greater than zero.")

    semi_minor_axis_length_m = fresnel_zone_radius(A, B, 1, frequency_hz, semi_major_axis_length_m)
    
    if semi_minor_axis_length_m <= 0:
        raise ValueError("Semi-minor axis length must be greater than zero.")
    
    volume_m3 = (4 / 3) * math.pi * semi_minor_axis_length_m**2 * semi_major_axis_length_m
    
    return volume_m3
    

def ring_centres(
    A: np.ndarray, B: np.ndarray, num_of_rings: int
) -> np.ndarray:
    """Calculates the coordinates of the centres of the rings along the line of sight (LOS)."""
    ring_centres_x: np.ndarray = np.linspace(A[0], B[0], num_of_rings + 2)[1:-1]
    ring_centres_y: np.ndarray = np.linspace(A[1], B[1], num_of_rings + 2)[1:-1]
    ring_centres_z: np.ndarray = np.linspace(A[2], B[2], num_of_rings + 2)[1:-1]

    ring_centres = np.column_stack((ring_centres_x, ring_centres_y, ring_centres_z))

    return ring_centres


def rotate_vector(
    v: np.ndarray, angle_degrees: float, axis: np.ndarray
) -> np.ndarray:
    """Rotates vector v around the given axis by the specified angle in degrees."""
    angle_radians = np.radians(angle_degrees)
    axis = axis / np.linalg.norm(axis)  # Normalize the rotation axis
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    
    rotated_v = (  # Rodrigues' rotation formula
        v * cos_theta + np.cross(axis, v) * sin_theta + axis * np.dot(axis, v) * (1 - cos_theta)
    )
    
    return rotated_v

def fresnel_zone_points(
    A: np.ndarray, 
    B: np.ndarray, 
    num_of_rings: int, 
    num_of_points_per_ring: int,
    frequency_hz: float
) -> np.ndarray:
    """Calculates the convex hull of the points on the surface of the Fresnel zone."""
    points = [A, B]
    
    # Set the angles in degrees
    angles = np.linspace(0, 360, num_of_points_per_ring, endpoint=False)
    angles += 180  # Rotate the angles by 180 degrees
    angles = angles % 360  # Ensure angles are within [0, 360) range
    
    # Set the heading vector and the up vector
    d_hat = heading_vector(A, B)
    up = np.array([0, 0, 1])  # Up vector in the z-direction
    w = up - np.dot(up, d_hat) * d_hat  # Project up vector onto the plane perpendicular to d_hat
    
    for ring_centre in ring_centres(A, B, num_of_rings):
        
        d1_m = distance(A, ring_centre)
        radius_m = fresnel_zone_radius(A, B, 1, frequency_hz, d1_m)
        v: np.ndarray = radius_m * w / np.linalg.norm(w) 
        
        for angle in angles:
            
            rotated_v = rotate_vector(v, angle, d_hat)
            point_on_ring = np.array(ring_centre + rotated_v)
            points.append(point_on_ring) 
            
    return np.array(points)

def save_points_to_csv(
    points: np.ndarray, filename: str
) -> None:
    """Saves the points to a CSV file."""
    np.savetxt(f"data/{filename}", points, delimiter=",", header="Easting,Northing,Z", comments="")