"""This module provides the physics needed for the maxwell probability calculations"""

import numpy as np


def maxwell(speed, kbt, mass=1.0):
    """Calculates the normalized maxwell boltzmann distribution probability

    The normalization constant of the distribution is mass/kbt

    Args:
        speed (float) : The speed of the ball
        kbt (float) : The thermal energy of the system
        mass (float) : The mass of the ball

    Returns:
        maxwell_prob (float) : The maxwell boltzmann distribution probability
    """
    if isinstance(speed, list):
        speed = np.array(speed, dtype=float)
    maxwell_prob = mass / kbt * speed * np.exp(-mass * speed * speed / (2 * kbt))
    return maxwell_prob
