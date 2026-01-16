"""
Tests for functionality checks in class SolveDiffusion2D
"""
import numpy as np
from diffusion2d import SolveDiffusion2D


def test_initialize_physical_parameters():
    """
    Integration test: initialize_domain + initialize_physical_parameters
    Check computed dt matches the expected formula along with expected values from previous funcations.
    """
    solver = SolveDiffusion2D()

    w, h = 12.0, 7.5
    dx, dy = 0.3, 0.5
    D = 3.0
    T_cold, T_hot = 250.0, 900.0

    solver.initialize_domain(w=w, h=h, dx=dx, dy=dy)
    solver.initialize_physical_parameters(d=D, T_cold=T_cold, T_hot=T_hot)

    expected_nx = int(w / dx)
    expected_ny = int(h / dy)

    dx2 = dx * dx
    dy2 = dy * dy
    expected_dt = dx2 * dy2 / (2.0 * D * (dx2 + dy2))

    assert solver.w == w
    assert solver.h == h
    assert solver.dx == dx
    assert solver.dy == dy
    assert solver.nx == expected_nx
    assert solver.ny == expected_ny
    assert solver.T_cold == T_cold
    assert solver.T_hot == T_hot
    assert solver.D == D
    assert solver.dt == expected_dt
  
       
def test_set_initial_condition():
    """
    Integration test: initialize_domain + initialize_physical_parameters + set_initial_condition
    Compute expected u array and compare along with expected values from previous funcations.
    """
    solver = SolveDiffusion2D()

    w, h = 10.0, 5.0
    dx, dy = 1.0, 0.5
    D = 4.0
    T_cold, T_hot = 300.0, 700.0

    solver.initialize_domain(w=w, h=h, dx=dx, dy=dy)
    solver.initialize_physical_parameters(d=D, T_cold=T_cold, T_hot=T_hot)
    u = solver.set_initial_condition()

    expected_nx = int(w / dx)
    expected_ny = int(h / dy)

    dx2 = dx * dx
    dy2 = dy * dy
    expected_dt = dx2 * dy2 / (2.0 * D * (dx2 + dy2))
    expected_u = T_cold * np.ones((expected_nx, expected_ny))

    r, cx, cy = 2, 5, 5
    r2 = r ** 2
    for i in range(expected_nx):
        for j in range(expected_ny):
            p2 = (i * dx - cx) ** 2 + (j * dy - cy) ** 2
            if p2 < r2:
                expected_u[i, j] = T_hot

    assert solver.w == w
    assert solver.h == h
    assert solver.dx == dx
    assert solver.dy == dy
    assert solver.nx == expected_nx
    assert solver.ny == expected_ny
    assert solver.T_cold == T_cold
    assert solver.T_hot == T_hot
    assert solver.D == D
    assert solver.dt == expected_dt
    assert np.array_equal(u, expected_u)