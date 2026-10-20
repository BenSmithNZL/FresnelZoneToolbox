import numpy as np
import pytest

from core import distance, heading_vector


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