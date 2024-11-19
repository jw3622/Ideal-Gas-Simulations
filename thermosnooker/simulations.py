"""The module runs the simulations based on the balls module"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from thermosnooker import balls


class Simulation:
    """Base class of different simulation scenarios"""

    def next_collision(self):
        """Base class for performing the next collision

        Raises:
            NotImplementedError : the next collision method should be implemented \
            in the derived class

        """
        raise NotImplementedError(
            "next_collision() needs to be implemented in derived classes"
        )

    def setup_figure(self):
        """Base class for setting up the figure for visualizing simulations

        Raises:
            NotImplementedError : the setup figure method should be implemented \
            in the derived class

        """
        raise NotImplementedError(
            "setup_figure() needs to be implemented in derived classes"
        )

    def run(self, num_collisions, animate=False, pause_time=0.001):
        """Running the simulation

        Args:
            num_collisions (int) : The number of collisions simulated
            animate (Bool) : Choose to animate the simulation or not
            pause_time (float) : The time between each simulation frames
        """
        if animate:
            self.setup_figure()
        for _ in range(num_collisions):
            self.next_collision()

            if animate:
                plt.pause(pause_time)
        if animate:
            plt.show()


class SingleBallSimulation(Simulation):
    """Simulating single ball container collisions"""

    def __init__(self, container, ball):
        """Initializes the container and ball object for single ball simulation

        Args:
            container (class) : The container for the ball in the simulation
            ball (class) : The ball used in the simulation
        """
        self.__container = container
        self.__ball = ball

    def container(self):
        """Access method for the container

        Returns:
            container (class) : The container for the ball in the simulation
        """
        return self.__container

    def ball(self):
        """Access method for the ball

        Returns:
            ball (class) : The ball in collision simulation
        """
        return self.__ball

    def setup_figure(self):
        """Setting up the animation canvas"""
        rad = self.container().radius()
        fig = plt.figure(figsize=(3, 3))
        ax = plt.axes(xlim=(-rad, rad), ylim=(-rad, rad))
        ax.add_artist(self.container().patch())
        ax.add_patch(self.ball().patch())

    def next_collision(self):
        """Performing the next collision

        Finds the time it takes until the next collision, then forwards the position\
        of the ball towards that time, and updating the velocity of the ball and the \
        container after the collision. 
        """
        # Calculate time for collision
        container = self.container()
        ball = self.ball()

        time = ball.time_to_collision(container)

        # Advance Balls in time
        ball.move(time)

        # Collide the ball and container
        container.collide(ball)


class MultiBallSimulation(Simulation):
    """Simulating multiple balls in the thermosnooker"""

    def __init__(
        self,
        c_radius=10.0,
        b_radius=1.0,
        b_speed=10.0,
        b_mass=1.0,
        rmax=8.0,
        nrings=3,
        multi=6,
    ):
        """Initializing the multi ballsimulation

        Args:
            c_radius (float) : Radius of container
            b_radius (float) : Radius of ball
            b_speed (float) : Speed of the Ball
            b_mass (float) : Mass of the ball
            rmax (float) : Maximum radius of the ring
            nrings (int) : Numebr of rings
            multi (int) : Multiplier for the number of balls of each ring
        """
        # Container
        self.__container = balls.Container(c_radius)
        # Balls
        self.__pos = self.rtrings(rmax, nrings, multi)
        self.__balls = []

        for pos in self.__pos:
            theta = np.random.uniform(0, 2 * np.pi)
            vel = self.random_velocity_calculator(b_speed, theta)
            self.__balls.append(
                balls.Ball(pos=pos, vel=vel, radius=b_radius, mass=b_mass)
            )

        self.__tot_time = 0.0

    def rtrings(self, rmax, nrings, multi):
        """Creating rings of evenly distributed position of balls

        Args:
            rmax (float) : The maximum radius of the ring
            nrings (float) : The number of rings needed
            multi (float) : The number of points in each ring calculated by multi * nrings = npoints

        Yields:
            The positions indicated by r and theta values
        """
        rmax = int(rmax)

        for n in range(0, nrings + 1):
            if n == 0:
                yield [0.0, 0.0]
            npoints = n * multi
            dr = rmax / nrings

            for m in range(0, npoints):
                r = dr * n
                theta = (2 * np.pi / npoints) * m
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                yield [x, y]

    def random_velocity_calculator(self, magnitude, angle):
        """Generates the random velocity for each ball

        Args:
            magnitude (float) : The magnitude of the velocity generated
            angle (float) : The random angle of the velocity generated

        Returns:
            velocity (list) : A list containing the x and y velocity
        """
        x_velocity = magnitude * np.cos(angle)
        y_velocity = magnitude * np.sin(angle)
        return [x_velocity, y_velocity]

    def container(self):
        """Accessor method of the container object

        Returns:
            container (object) : The container object initialized
        """
        return self.__container

    def balls(self):
        """Accessor method of the ball objects

        Returns:
            balls (list) : A list of balls that has been initialized
        """
        return self.__balls

    def balls_pos(self):
        """Getting the position of the balls as a list

        Returns:
            ball_pos (array) : A list of the ball positions, which are 2d positions
        """
        balls_pos = []
        for ball in self.balls():
            balls_pos.append(ball.pos())

        return np.array(balls_pos)

    def setup_figure(self):
        """Setting up the animation canvas"""
        rad = self.container().radius()
        fig = plt.figure(figsize=(3, 3))
        ax = plt.axes(xlim=(-rad, rad), ylim=(-rad, rad))
        ax.add_artist(self.container().patch())
        for ball in self.balls():
            ax.add_patch(ball.patch())

    def next_collision(self):
        """Finding the next collision between balls and container

        The method presets the minimal time and next event, and from calculating\
        the time each ball has to collide with the container and all other balls\
        we update the minimal time each time we find a smaller time to collision\
        after running all of the time to collision comparisons and obtaining a minimal\
        time, we determine if its a ball container or a ball ball collision and \
        identify which ball or pair of balls is involved in the collision, \
        to step forward that amount of time and perform the collision. 

        Raises:
            RuntimeError: When there is no collisions detected
        """

        min_time = float(np.inf)
        next_event = None
        # Calculate time for collision of each ball with the container
        for ball in self.balls():
            time = balls.Ball.time_to_collision(self.container(), ball)
            if time is not None and time < min_time:
                min_time = time
                next_event = ("container", ball)  # Indexing container ball collision

        # Calculate time for collision of each ball with the other balls
        for i, ball_1 in enumerate(self.balls()):
            for j, ball_2 in enumerate(self.balls()):
                if i < j:
                    time = ball_1.time_to_collision(ball_2)
                    if time is not None and time < min_time:
                        min_time = time
                        next_event = ("balls", ball_1, ball_2)
        for ball in self.balls():
            balls.Ball.move(ball, min_time)

        if next_event is None:
            raise RuntimeError("No collision detected")
        if next_event[0] == "container":
            ball = next_event[1]
            balls.Ball.collide(self.container(), ball)
        elif next_event[0] == "balls":
            ball_1 = next_event[1]
            ball_2 = next_event[2]
            balls.Ball.collide(ball_1, ball_2)

        self.__tot_time += min_time

    def kinetic_energy(self):
        """Calculating and obtaining the total kinetic energy of the system

        Returns:
            Kinetic Energy (float) : The total kinetic energy of the system
        """

        ke_tot = 0.0
        for ball in self.balls():
            ke_tot += 0.5 * ball.mass() * np.dot(ball.vel(), ball.vel())

        ke_tot += (
            0.5
            * self.container().mass()
            * np.dot(self.container().vel(), self.container().vel())
        )
        return ke_tot

    def momentum(self):
        """Calculating and obtaining the total momentum of the system

        Returns:
            Momentum (float) : The total momentum of the system
        """

        momentum = [0.0, 0.0]
        for ball in self.balls():
            momentum += ball.mass() * ball.vel()
        momentum = np.array(momentum)
        momentum += self.container().momentum()
        return momentum

    def time(self):
        """Accessor method for the total time that the simulation has been running

        Returns:
            total time (float) : The total time that the simulation has been running
        """
        return self.__tot_time

    def pressure(self):
        """Calculating and obtaining pressure

        Returns:
            pressure (float) : The pressure of the system
        """
        cont = self.container()
        pressure = cont.dp_tot() / (cont.surface_area() * self.time())

        return pressure

    def run(self, num_collisions, animate=False, pause_time=0.001):
        """Running the simulation

        Args:
            num_collisions (int) : The number of collisions simulated
            animate (Bool) : Choose to animate the simulation or not
            pause_time (float) : The time between each simulation frames

        Returns:
            times (list) : The time elapsed in the simulation
            ke_ratios (list) : The ratio of kinetic energy with initial kinetic energy
            momentum_x_ratios (list) : The ratio of momentum with initial momentum in x direction
            momentum_y_ratios (list) : The ratio of momentum with initial momentum in y direction
            pressures (list) : The pressure of the system
        """
        times = []
        ke_ratios = []
        momentum_x_ratios = []
        momentum_y_ratios = []
        pressures = []

        initial_ke = self.kinetic_energy()
        initial_momentum = self.momentum()
        if animate:
            self.setup_figure()

        for _ in range(num_collisions):
            self.next_collision()
            times.append(self.time())
            ke_ratios.append(self.kinetic_energy() / initial_ke)
            momentum = self.momentum()
            momentum_x_ratios.append(momentum[0] / initial_momentum[0])
            momentum_y_ratios.append(momentum[1] / initial_momentum[1])
            pressures.append(self.pressure())
            print(f"skr{_}")

            if animate:
                plt.pause(pause_time)
        if animate:
            plt.show()

        return times, ke_ratios, momentum_x_ratios, momentum_y_ratios, pressures

    def t_equipartition(self):
        """Calculates the temperature from equipartition

        Returns:
            equipartition_temperature (float) : The temperature of the system from equipartition
        """
        avg_ke = self.kinetic_energy() / len(self.balls())
        equipartition_temperature = avg_ke / sp.constants.Boltzmann

        return equipartition_temperature

    def t_ideal(self):
        """Calculates the temperature from PV = NKbT

        Returns:
            idealgas_temperature (float) : The temperature of the system from ideal gas calculations
        """
        pressure = self.pressure()
        volume = self.container().volume()
        nballs = len(self.balls())
        idealgas_temperature = pressure * volume / (nballs * sp.constants.Boltzmann)

        return idealgas_temperature

    def speeds(self):
        """Calculates the speed of the balls in the simulation

        Returns:
            speeds (list) : A lists of the speed of all the balls in the systen
        """
        speeds = []
        for ball in self.balls():
            speeds.append(np.linalg.norm(ball.vel()))

        return speeds

    def p_ideal(self):
        """Calculates the ideal pressure through IGL

        Returns:
            pressure (float) : The pressure of the system from ideal gas calculations
        """
        volume = self.container().volume()
        nballs = len(self.balls())
        t_equipartition = self.t_equipartition()

        pressure = nballs * sp.constants.Boltzmann * t_equipartition / volume

        return pressure
