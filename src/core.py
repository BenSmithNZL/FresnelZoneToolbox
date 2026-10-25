"""Core functions for Fresnel zone calculations."""
import math

import numpy as np

C = 299_792_458  # speed of light in meters per second

# ------ Validation functions ------
def _validate_point(
    point: np.ndarray,
    name: str
) -> np.ndarray:
    """Validate that the input point is a 3D numpy array."""
    if not isinstance(point, np.ndarray):
        raise TypeError(f"{name} must be a numpy array.")
    
    if point.shape != (3,):
        raise ValueError(
            f"{name} must be a 3D point with shape (3,) but got shape "
            f"{point.shape}."
        )

    return point


def _validate_float(
    value: float,
    name: str,
    lower_bound: int,
    eq: bool
) -> float:
    """Validate that the input value is a float greaterthan (or equal to) the
    specified lower bound.
    """
    if not isinstance(value, float):
        raise TypeError(f"{name} must be a float.")
    
    if eq:
        if value < lower_bound:
            raise ValueError(f"{name} must be greater than "
                            f"or equal to {lower_bound}.")
    else:
        if value <= lower_bound:
            raise ValueError(f"{name} must be greater than {lower_bound}.")

    return value

def _validate_int(
    value: int,
    name: str,
    lower_bound: int,
    eq: bool
) -> int:
    """Validate that the input value is an int greater than (or equal to) the
    specified lower bound.
    """
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    
    if eq:
        if value < lower_bound:
            raise ValueError(
                f"{name} must be greater than or equal to {lower_bound}."
            )
    else:
        if value <= lower_bound:
                raise ValueError(f"{name} must be greater than {lower_bound}.")

    return value        


# ----- Core functions -----
def distance(
    a: np.ndarray, b: np.ndarray
) -> float:
    """Calculate the Euclidean distance between points a and b."""
    a, b = _validate_point(a, "a"), _validate_point(b, "b")

    return math.sqrt(
        (b[0] - a[0]) ** 2 
        + (b[1] - a[1]) ** 2 
        + (b[2] - a[2]) ** 2
    )


def heading_vector(
    a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Calculate the normalised vector pointing from point a to point b."""
    a, b = _validate_point(a, "a"), _validate_point(b, "b")
    
    heading_vector = (np.array(b) - np.array(a)) / distance(a, b)
    
    return heading_vector


def fresnel_zone_radius(
    a: np.ndarray, b: np.ndarray, n: int, frequency_hz: float, d1_m: float
) -> float:
    """Calculate the radius of the nth Fresnelzone at a given distance from
    point a.
    """
    a, b = _validate_point(a, "a"), _validate_point(b, "b")
    n = _validate_int(n, "n", 1, eq=True)
    frequency_hz = _validate_float(frequency_hz, "frequency_hz", 0, eq=False)
    d1_m = _validate_float(d1_m, "d1_m", 0, eq=True)
    
    d_m = distance(a, b)
    
    if d_m <= d1_m:
        raise ValueError(
            f"Distance from point a to the point of interest must be "
            f"less than the distance between points a and b ({d_m} m)."
        )
    
    d2_m = d_m - d1_m
    wavelength_m = C / frequency_hz
    radius_m = math.sqrt(n * wavelength_m * d1_m * d2_m / d_m)
    
    return radius_m


def theoretical_volume(
    a: np.ndarray, b: np.ndarray, frequency_hz: float
) -> float:
    """Calculate the volume of the ellipsoid with foci at points a and b."""
    a, b = _validate_point(a, "a"), _validate_point(b, "b")
    frequency_hz = _validate_float(frequency_hz, "frequency_hz", 0, eq=False)
    
    semi_major_axis_length_m = distance(a, b) / 2
    
    if semi_major_axis_length_m <= 0:
        raise ValueError("Semi-major axis length must be greater than zero.")

    semi_minor_axis_length_m = fresnel_zone_radius(
        a, b, 1, frequency_hz, semi_major_axis_length_m
    )
    
    if semi_minor_axis_length_m <= 0:
        raise ValueError("Semi-minor axis length must be greater than zero.")
    
    volume_m3 = (
        (4 / 3) 
        * math.pi 
        * semi_minor_axis_length_m**2 
        * semi_major_axis_length_m
    )
    
    return volume_m3
    

def ring_centres(
    a: np.ndarray, b: np.ndarray, num_of_rings: int
) -> np.ndarray:
    """Calculate the coordinates of the centres of the rings along the line of
    sight (LOS).
    """
    a, b = _validate_point(a, "a"), _validate_point(b, "b")
    num_of_rings = _validate_int(num_of_rings, "num_of_rings", 1, eq=True)
    
    ring_centres_x= np.linspace(a[0], b[0], num_of_rings + 2)[1:-1]
    ring_centres_y= np.linspace(a[1], b[1], num_of_rings + 2)[1:-1]
    ring_centres_z= np.linspace(a[2], b[2], num_of_rings + 2)[1:-1]

    ring_centres = np.column_stack(
        (ring_centres_x, ring_centres_y, ring_centres_z)
    )

    return ring_centres


def rotate_vector(
    v: np.ndarray, angle_degrees: float, axis: np.ndarray
) -> np.ndarray:
    """Rotates vector v around the givenaxis by the specified angle in
    degrees.
    """
    v, axis = _validate_point(v, "v"), _validate_point(axis, "axis")
    angle_degrees = _validate_float(angle_degrees, "angle_degrees", 0, eq=True)
    
    angle_radians = np.radians(angle_degrees)
    axis = axis / np.linalg.norm(axis)  # Normalize the rotation axis
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    
    rotated_v = (  # Rodrigues' rotation formula
        v * cos_theta 
        + np.cross(axis, v) * sin_theta 
        + axis * np.dot(axis, v) * (1 - cos_theta)
    )
    
    return rotated_v

def fresnel_zone_points(
    a: np.ndarray, 
    b: np.ndarray, 
    num_of_rings: int, 
    num_of_points_per_ring: int,
    frequency_hz: float
) -> np.ndarray:
    """Return the points on the surface of the
    Fresnel zone defined by points a and b.
    """
    a, b = _validate_point(a, "a"), _validate_point(b, "b")
    num_of_rings = _validate_int(num_of_rings, "num_of_rings", 3, eq=True)
    num_of_points_per_ring = _validate_int(
        num_of_points_per_ring, "num_of_points_per_ring", 3, eq=True
    )
    frequency_hz = _validate_float(frequency_hz, "frequency_hz", 0, eq=False)

    points = [a, b]  # Start with the endpoints of the Fresnel zone
    
    # Set the angles in degrees
    angles = np.linspace(0, 360, num_of_points_per_ring, endpoint=False)
    angles += 180  # Rotate the angles by 180 degrees
    angles = angles % 360  # Ensure angles are within [0, 360) range
    
    # Set the heading vector and the up vector
    d_hat = heading_vector(a, b)
    up = np.array([0, 0, 1])  # Up vector in the z-direction
    w = up - np.dot(up, d_hat) * d_hat  # Vector perpendicular to d_hat
    
    for ring_centre in ring_centres(a, b, num_of_rings):
        
        d1_m = distance(a, ring_centre)
        radius_m = fresnel_zone_radius(a, b, 1, frequency_hz, d1_m)
        v: np.ndarray = radius_m * w / np.linalg.norm(w) 
        
        for angle in angles:
            
            rotated_v = rotate_vector(v, angle, d_hat)
            point_on_ring = np.array(ring_centre + rotated_v)
            points.append(point_on_ring) 
            
    return np.array(points)

def save_points_to_csv(
    points: np.ndarray, filename: str
) -> None:
    """Save the points to a CSV file."""
    points = _validate_point(points, "points")
    np.savetxt(
        f"data/{filename}", 
        points, delimiter=",", 
        header="Easting,Northing,Z", 
        comments=""
    )