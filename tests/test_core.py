import numpy as np
import pytest

from core import distance, heading_vector, fresnel_zone_radius, ring_centres


# ---------------- distance -------------------


def test_distance_3_4_5_triangle():
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([3.0, 4.0, 0.0])
    assert distance(A, B) == pytest.approx(5.0)


def test_distance_is_symmetric():
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([3.0, 4.0, 0.0])
    assert distance(A, B) == pytest.approx(distance(B, A))
    
    
def test_distance_is_zero_for_same_point():
    A = np.array([3.0, 4.0, 0.0])
    assert distance(A, A) == pytest.approx(0.0)


# ---------------- heading_vector -------------------


def test_heading_vector_is_unit_length():
    A = np.array([1.0, 1.0, 1.0])
    B = np.array([4.0, 5.0, 1.0])
    h = heading_vector(A, B)
    assert np.linalg.norm(h) == pytest.approx(1.0)
    
    
def test_heading_vector_points_to_b():
    A = np.array([1.0, 1.0, 1.0])
    B = np.array([4.0, 5.0, 1.0])
    h = heading_vector(A, B)
    
    assert h == pytest.approx([0.6, 0.8, 0.0])
    
    
def test_heading_vector_is_anti_parallel_to_reverse_heading_vector():
    A = np.array([1.0, 1.0, 1.0])
    B = np.array([4.0, 5.0, 1.0])
    h_1 = heading_vector(A, B)
    h_2 = heading_vector(B, A)
    
    assert h_1 == pytest.approx(-h_2)
    

# ---------------- fresnel_zone_radius -------------------


@pytest.mark.parametrize("bad_n", [0, -1, -5])
def test_non_positive_n_raises(bad_n):
    A = np.array([1000.0, 1000.0, 1000.0])
    B = np.array([4000.0, 5000.0, 1000.0])
    with pytest.raises(ValueError, match="positive integer"):
        fresnel_zone_radius(A, B, bad_n, 1e9, 300.0)


@pytest.mark.parametrize("bad_n", [1.5, 2.0, "1", None])
def test_non_integer_n_raises(bad_n):
    A = np.array([1000.0, 1000.0, 1000.0])
    B = np.array([4000.0, 5000.0, 1000.0])
    with pytest.raises(TypeError, match="n must be an integer"):
        fresnel_zone_radius(A, B, bad_n, 1e9, 300.0)


@pytest.mark.parametrize("bad_freq", [0, 0.0, -1.0, -1e9])
def test_non_positive_frequency_raises(bad_freq):
    A = np.array([1.0, 1.0, 1.0])
    B = np.array([4.0, 5.0, 1.0])
    with pytest.raises(ValueError, match="Frequency must be a positive"):
        fresnel_zone_radius(A, B, 1, bad_freq, 300.0)


def test_non_positive_distance_raises():
    A = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="Distance between points"):
        fresnel_zone_radius(A, A, 1, 6e9, 300.0)
   
        
def test_d1_greater_than_distance():
    A = np.array([0, 0, 0])
    B = np.array([1, 1, 1])
    with pytest.raises(ValueError, match="Distance from point A to the point of interest must be"):
        fresnel_zone_radius(A, B, 1, 6e9, 300.0)
        

# ---------------- ring_centres -------------------
@pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
def test_output_shape(n):
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([1.0, 2.0, 3.0])
    result = ring_centres(A, B, n)
    assert result.shape == (n, 3)

def test_returns_ndarray():
    result = ring_centres(np.zeros(3), np.ones(3), 3)
    assert isinstance(result, np.ndarray)

def test_dtype_is_float():
    # Integer inputs should still produce float output
    A = np.array([0, 0, 0])
    B = np.array([4, 4, 4])
    result = ring_centres(A, B, 3)
    assert np.issubdtype(result.dtype, np.floating)