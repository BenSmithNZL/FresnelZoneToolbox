"""Test functions contained in core.py."""

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from src.core import (
    distance,
    fresnel_zone_points,
    fresnel_zone_radius,
    heading_vector,
    ring_centres,
    theoretical_volume,
)

# ---------------- distance -------------------


def test_distance_3_4_5_triangle() -> None:
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([3.0, 4.0, 0.0])
    assert distance(a, b) == pytest.approx(5.0)


def test_distance_is_symmetric() -> None:
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([3.0, 4.0, 0.0])
    assert distance(a, b) == pytest.approx(distance(b, a))


def test_distance_is_zero_for_same_point() -> None:
    a = np.array([3.0, 4.0, 0.0])
    assert distance(a, a) == pytest.approx(0.0)


# ---------------- heading_vector -------------------


def test_heading_vector_is_unit_length() -> None:
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([4.0, 5.0, 1.0])
    h = heading_vector(a, b)
    assert np.linalg.norm(h) == pytest.approx(1.0)
    
    
def test_heading_vector_points_to_b() -> None:
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([4.0, 5.0, 1.0])
    h = heading_vector(a, b)
    
    assert h == pytest.approx([0.6, 0.8, 0.0])
    
    
def test_heading_vector_is_anti_parallel_to_reverse_heading_vector() -> None:
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([4.0, 5.0, 1.0])
    h_1 = heading_vector(a, b)
    h_2 = heading_vector(b, a)
    
    assert h_1 == pytest.approx(-h_2)
    

# ---------------- fresnel_zone_radius -------------------


@pytest.mark.parametrize("bad_n", [0, -1, -5])
def test_non_positive_n_raises(bad_n) -> None:
    a = np.array([1000.0, 1000.0, 1000.0])
    b = np.array([4000.0, 5000.0, 1000.0])
    with pytest.raises(
        ValueError, 
        match=r"n must be greater than or equal to 1."
    ):
        fresnel_zone_radius(a, b, bad_n, 1e9, 300.0)



@pytest.mark.parametrize("bad_n", [1.5, 2.0, "1", None])
def test_non_integer_n_raises(bad_n) -> None:
    a = np.array([1000.0, 1000.0, 1000.0])
    b = np.array([4000.0, 5000.0, 1000.0])
    with pytest.raises(
        TypeError, 
        match="n must be an integer"
    ):
        fresnel_zone_radius(a, b, bad_n, 1e9, 300.0)


@pytest.mark.parametrize("bad_freq", [0.0, -1.0, -10.21, -1e9])
def test_non_positive_frequency_raises(bad_freq) -> None:
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([4.0, 5.0, 1.0])
    with pytest.raises(
        ValueError, 
        match=r"frequency_hz must be greater than 0."
    ):
        fresnel_zone_radius(a, b, 1, bad_freq, 300.0)


def test_non_positive_distance_raises() -> None:
    a = np.array([1.0, 1.0, 1.0])
    with pytest.raises(
        ValueError, 
        match=r"Distance from point a "
    ):
        fresnel_zone_radius(a, a, 1, 6e9, 300.0)
   
        
def test_d1_greater_than_distance() -> None:
    a = np.array([0, 0, 0])
    b = np.array([1, 1, 1])
    with pytest.raises(
        ValueError, 
        match=r"Distance from point a to the point of interest must be"      
    ):
        fresnel_zone_radius(a, b, 1, 6e9, 300.0)
        

# ---------------- ring_centres -------------------


@pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
def test_output_shape(n) -> None:
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 2.0, 3.0])
    result = ring_centres(a, b, n)
    assert result.shape == (n, 3)

def test_returns_ndarray() -> None:
    result = ring_centres(np.zeros(3), np.ones(3), 3)
    assert isinstance(result, np.ndarray)

def test_dtype_is_float() -> None:
    # Integer inputs should still produce float output
    a = np.array([0, 0, 0])
    b = np.array([4, 4, 4])
    result = ring_centres(a, b, 3)
    assert np.issubdtype(result.dtype, np.floating)
    
# ---------------- Test convergence -------------------
def test_error_decreases_with_number_of_rings() -> None:
    a = np.array([0, 0, 100])
    b = np.array([10, 25, 120])
    frequency_hz = 6e9
    
    errors = []
    
    theoretical_volume_m3 = theoretical_volume(
        a=a,
        b=b,
        frequency_hz=frequency_hz
    )
    
    for num_of_rings in [5, 25, 100]:

        points = fresnel_zone_points(
            a=a, 
            b=b, 
            num_of_rings=num_of_rings, 
            num_of_points_per_ring=3, 
            frequency_hz=frequency_hz
        )

        hull = ConvexHull(points=points)
        error = abs(theoretical_volume_m3 - hull.volume)
        errors.append(error)

    assert errors[0] > errors[1] > errors[2]
    
    
def test_error_decreases_with_number_of_points_per_ring() -> None:
    a = np.array([0, 0, 100])
    b = np.array([10, 25, 120])
    frequency_hz = 6e9
    
    errors = []
    
    theoretical_volume_m3 = theoretical_volume(
        a=a,
        b=b,
        frequency_hz=frequency_hz
    )
    
    for number_of_points_per_ring in [5, 25, 100]:

        points = fresnel_zone_points(
            a=a, 
            b=b, 
            num_of_rings=5, 
            num_of_points_per_ring=number_of_points_per_ring, 
            frequency_hz=frequency_hz
        )

        hull = ConvexHull(points=points)
        error = abs(theoretical_volume_m3 - hull.volume)
        errors.append(error)

    assert errors[0] > errors[1] > errors[2]