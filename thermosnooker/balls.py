"""
This module provides the initialization and modification of values of balls used\
in the simulation
"""

import matplotlib.pyplot as plt
import numpy as np


class Ball:
    """
    Base class for creating the balls in the hard sphere simulation
    """

    def __init__(self, pos=None, vel=None, radius=1.0, mass=1.0):
        """Defining the propperties of the balls in the thermalsnooker

        This function initiates a ball used in the simulation of the\
        thermosnooker, giving them properties that will later be used in the\
        simulation.

        Args:
            pos (ndarray) : The 2D position of the ball.
            vel (ndarray) : The 2D velocity of the ball.
            radius (float) : The radius of the ball.
            mass (float) : The mass of the ball.
        
        Raises:
            TypeError: The position of the ball needs to be a list or an array
            ValueError: The position needs to have 2 positional arguments 
            TypeError: The velocity of the ball needs to be a list or an array
            ValueError: The velocity needs to have 2 positional arguments
        """

        if pos is None:
            pos = [0.0, 0.0]
        if vel is None:
            vel = [1.0, 0.0]
        # Checking for invalid inputs
        if not isinstance(pos, np.ndarray) and not isinstance(pos, list):
            raise TypeError("The position must be a list or an array")
        if len(pos) != 2:
            raise ValueError("The position must have 2 positional arguments")
        if not isinstance(vel, np.ndarray) and not isinstance(vel, list):
            raise TypeError("The velocity must be a list or an array")
        if len(vel) != 2:
            raise ValueError("The velocity must have 2 positional arguments")

        self.__pos = np.float64(np.array(pos))
        self.__vel = np.float64(np.array(vel))
        self.__radius = radius
        self.__mass = mass
        self.__momentum = np.array([0.0, 0.0])
        if not isinstance(self, Container):
            self.__patch = plt.Circle(pos, radius)  # type: ignore
        if isinstance(self, Container):
            self.__patch = plt.Circle(pos, radius, fill=False, ls="solid")  # type: ignore

    def pos(self):
        """Prints the 2D current position of the ball.

        The function access the current position attribute of the ball

        Returns:
            numpy array: The list containing two floats - the x and y position of the
            ball.
        """
        self.__pos = np.array(self.__pos)
        return self.__pos

    def radius(self):
        """Prints the radius of the ball

        Returns:
            float: The radius of the ball
        """
        return self.__radius

    def mass(self):
        """Prints the mass of the ball

        Returns:
            mass (float): The mass of the ball
        """
        return self.__mass

    def vel(self):
        """Prints the velocity of the ball

        Returns:
            numpy array: a list of the x and y direction velocity of the ball
        """
        return self.__vel

    def set_vel(self, vel):
        """Sets the velocity of the ball

        Args:
            vel (list) : The new velocity of the ball

        Raises:
            TypeError: The velocity of the ball needs to be a list or an array
            ValueError: The velocity needs to have 2 positional arguments
        """

        if not isinstance(vel, np.ndarray) and not isinstance(vel, list):
            raise TypeError("The velocity must be a list or an array")
        if len(vel) != 2:
            raise ValueError("The velocity must have 2 positional arguments")
        self.__vel = np.array(vel)

    def move(self, dt):
        """Moves the ball through a time period

        Uses equation r_new = r_current + v*t

        Args:
            dt (float) : The time period that the ball moves forward through
        """
        self.__pos += self.__vel * dt
        self.__patch.center += self.__vel * dt

    def patch(self):
        """Access the patch object

        Returns:
            patch (object) : The patch object representing the ball used for \
            animations.
        """
        return self.__patch

    def _quadratic_solver(self, a, b, c):
        """Solves a quadratic equation (ax^2+bx+c=0) for the collision time

        Utilizes the discriminant to check the type of the solution,\
        returns the real solution if possible. Then checksif the collision time\
        obtained is positive or negative, returning only the positive time solution.

        Args:
            a (float) : The coefficient of the squared term.
            b (float) : The coefficient of the linear term.
            c (float) : The constant/

        Returns:
            time (float) : The time it takes for the next collision to occur.
        """
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            time = None
        else:
            t1 = (-b + np.sqrt(discriminant)) / (2 * a)
            t2 = (-b - np.sqrt(discriminant)) / (2 * a)
            if np.isclose(t1, 0):
                t1 = 0
            if np.isclose(t2, 0):
                t2 = 0
            if t1 > 0 and t2 > 0:
                time = min(t1, t2)
            elif t1 > 0 or t2 > 0:
                time = max(t1, t2)
            else:
                time = None

        return time

    def time_to_collision(self, other):
        """Finding the time until the next collision happen

        The function uses the collision equation to determine the time it takes \
        for the next collision to happen, it utilizes the _quadratic_solver helper \
        function to solve a quadratic equation, and returns the shortest time it \
        takes for the collision to happen, given that a collision happens between \
        the two objects. 
        
        Args:
            other (object) : The other object that the ball is colliding with, \
            it can be a ball object or a wall object.

        Returns:
            t_internal (float) : The Ball Container collision time
            t_external (float) : The Ball Ball collision time

        Raises:
            TypeError: If the other is not a ball Object
        """

        r = self.pos() - other.pos()
        v = self.vel() - other.vel()
        pm = np.array([+1, -1])
        radius_sum, radius_diff = self.radius() + pm * other.radius()

        # Coefficients of the quadratic to be solved
        a = np.dot(v, v)
        b = 2 * np.dot(r, v)
        c_sum = np.dot(r, r) - radius_sum * radius_sum
        c_diff = np.dot(r, r) - radius_diff * radius_diff

        if not isinstance(self, Container) and not isinstance(other, Container):
            t_external = self._quadratic_solver(a, b, c_sum)
            return t_external

        if isinstance(other, Container) and not isinstance(self, Container):
            t_internal = self._quadratic_solver(a, b, c_diff)
            return t_internal

        if isinstance(self, Container) and not isinstance(other, Container):
            t_internal = self._quadratic_solver(a, b, c_diff)
            return t_internal

        raise TypeError("The other collision  must be a Ball or a Wall object")

    def collide(self, other):
        """Calculates the velocity and position after the collision

        The function uses the angle free representation of a two dimensional collision \
        with two moving objects, where np.linalg.norm calculates the distance between \
        the two objects.

        Args:
            other (object) : The object that the ball is colliding with

        Updates:
            self.__vel to the new velocity v1
            other.__vel to the new velocity v2
            self.__total_momentum added the new momentum recorded
        """

        m1, m2 = self.mass(), other.mass()
        u1, u2 = self.vel(), other.vel()
        r1, r2 = self.pos(), other.pos()

        # Calculating new velocity after collision
        v1 = u1 - 2 * m2 / (m1 + m2) * np.dot(u1 - u2, r1 - r2) / np.linalg.norm(
            r1 - r2
        ) ** 2 * (r1 - r2)
        v2 = u2 - 2 * m1 / (m1 + m2) * np.dot(u2 - u1, r2 - r1) / np.linalg.norm(
            r2 - r1
        ) ** 2 * (r2 - r1)

        if isinstance(self, Container):
            dp = np.linalg.norm(m1 * (v1 - u1))
            self.dp_tot_add(dp)
            self.__momentum += m1 * v1
            other.set_vel(np.round(v2, 4))
        else:
            self.set_vel(np.round(v1, 4))
            other.set_vel(np.round(v2, 4))

    def momentum(self):
        """Access the momentum of the container

        Returns:
            momentum (float) : The momentum of the container
        """
        return self.__momentum


class Container(Ball):
    """
    Provides the container of the balls
    """

    def __init__(self, radius=10.0, mass=10000000.0):
        """Defining the properties of the container of the balls

        Calls the initialization method of Ball class, gives the container radius, \
        and mass, and defaults the velocity and position to be [0,0].

        Args:
            radius (float) : The radius of the ball.
            mass (float) : The mass of the ball.
        """
        Ball.__init__(self, pos=None, vel=[0, 0], radius=radius, mass=mass)
        self.__total_momentum = 0.0

    def volume(self):
        """Finding the Volume of the container

        We are working in 2D, thus the volume is the area of the container.

        Returns:
            volume (float) : The volume of the container
        """
        volume = np.pi * self.radius() ** 2
        return volume

    def surface_area(self):
        """Finding the Volume of the container

        We are working in 2D, thus the surface area is the circumference of the \
        container.
        
        Returns:
            surface_area (float) : The volume of the container
        """
        surface_area = np.pi * 2 * self.radius()
        return surface_area

    def dp_tot_add(self, change_in_momentum):
        """Adding the new recorded change of momentum to the total momentum of the container

        Args:
            change_in_momentum (float) : The change in momentum of the container

        Updates:
            self.__total_momentum (float) : The new total momentum
        """
        self.__total_momentum += change_in_momentum

    def dp_tot(self):
        """Access the total momentum of the container

        Returns:
            total_momentum (float) : Total momentum of the Container
        """
        return self.__total_momentum
