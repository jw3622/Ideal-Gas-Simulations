"""Analysis Module."""

from datetime import datetime
from tkinter import font
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from thermosnooker import balls
from thermosnooker import simulations
from thermosnooker import physics


def task9():
    """
    Task 9.

    In this function, you should test your animation. To do this, create a container
    and ball as directed in the project brief. Create a SingleBallSimulation object from these
    and try running your animation. Ensure that this function returns the balls final position and
    velocity.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: The balls final position and velocity
    """
    c = balls.Container(radius=10.0)
    b = balls.Ball(pos=[-5, 0], vel=[1, 0], radius=1.0, mass=1.0)
    sbs = simulations.SingleBallSimulation(container=c, ball=b)

    sbs.run(20, True, 0.5)

    return sbs.ball().pos(), sbs.ball().vel()


def task10():
    """
    Task 10.

    In this function we shall test your MultiBallSimulation. Create an instance of this class using
    the default values described in the project brief and run the animation for 500 collisions.

    Watch the resulting animation carefully and make sure you aren't seeing errors like balls sticking
    together or escaping the container.
    """

    mbs = simulations.MultiBallSimulation()
    mbs.run(1000, True, 0.001)


def task11():
    """
    Task 11.

    In this function we shall be quantitatively checking that the balls aren't escaping or sticking.
    To do this, create the two histograms as directed in the project script. Ensure that these two
    histogram figures are returned.

    Returns:
        tuple[Figure, Firgure]: The histograms (distance from centre, inter-ball spacing).
    """
    mbs = simulations.MultiBallSimulation(nrings=10, b_radius=0.5)
    mbs.run(3000, False, 0.1)

    positions = mbs.balls_pos()
    distance_from_center = []
    for position in positions:
        distance_from_center.append(np.linalg.norm(position))

    center_distance_fig = plt.hist(distance_from_center, bins=15, edgecolor="black")
    plt.title("The number of balls at a given distance from the center")
    plt.xlabel("Distance from center (m)")
    plt.ylabel("Number of balls")
    plt.savefig("fig11_ballcentre", dpi=800)
    plt.show()

    distance_between_balls = []
    for i, position_a in enumerate(positions):
        for j, position_b in enumerate(positions):
            if i < j:
                distance_between_balls.append(np.linalg.norm(position_a - position_b))

    print(len(distance_between_balls))

    inter_ball_distance_fig = plt.hist(
        distance_between_balls, bins=20, edgecolor="black"
    )
    plt.title("The number of pairs of balls at a given inter-ball distance")
    plt.xlabel("Inter-ball distance (m)")
    plt.ylabel("Number of pairs of balls")
    plt.savefig("fig11_interball", dpi=800)
    plt.show()

    return center_distance_fig, inter_ball_distance_fig


def task12():
    """
    Task 12.

    In this function we shall check that the fundamental quantities of energy and momentum are conserved.
    Additionally we shall investigate the pressure evolution of the system. Ensure that the 4 figures
    outlined in the project script are returned.

    Returns:
        fig12_ke (figure) : The kinetic energy ratio evolution plot
        fig12_momx (figure) : The momentum x ratio evolution plot
        fig12_momy (figure) : The momentum y ratio evolution plot
        fig12_pt (figure) : The pressure evolution plot
    """
    mbs = simulations.MultiBallSimulation()
    times, ke_ratios, momentum_x_ratios, momentum_y_ratios, pressures = mbs.run(
        4000, False, 0.01
    )
    fig12_ke = plt.plot(times, ke_ratios, "-", label="KE ratio", linewidth=2)
    plt.ylim(0.99, 1.01)
    plt.title("Kinetic Energy Ratio Evolution")
    plt.xlabel("Time (s)")
    plt.ylabel("Kinetic Energy ratio")
    plt.legend()
    plt.savefig("fig12_ke", dpi=800)
    plt.show()

    fig12_momx = plt.plot(
        times, momentum_x_ratios, "-", label="Momentum x ratio", linewidth=2
    )
    plt.ylim(0.99, 1.01)
    plt.title("x Direction Momentum Ratio Evolution")
    plt.xlabel("Time (s)")
    plt.ylabel("Momentum x ratio")
    plt.legend()
    plt.savefig("fig12_momx", dpi=800)
    plt.show()

    fig12_momy = plt.plot(
        times, momentum_y_ratios, "-", label="Momentum y ratio", linewidth=2
    )
    plt.title("y Direction Momentum Ratio Evolution")
    plt.xlabel("Time (s)")
    plt.ylabel("Momentum y ratio")
    plt.legend()
    plt.ylim(0.99, 1.01)
    plt.savefig("fig12_momy", dpi=800)
    plt.show()

    fig12_pt = plt.plot(times, pressures, label="Pressure", linewidth=2)
    plt.title("Pressure Evolution")
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (Pa)")
    plt.legend()
    plt.savefig("fig12_pt", dpi=800)
    plt.show()

    return fig12_ke, fig12_momx, fig12_momy, fig12_pt


def task13():
    """
    Task 13.

    In this function we investigate how well our simulation reproduces the distributions of the IGL.
    Create the 3 figures directed by the project script, namely:
    1) PT plot
    2) PV plot
    3) PN plot
    Ensure that this function returns the three matplotlib figures.

    Returns:
        tuple[Figure, Figure, Figure]: The 3 requested figures: (PT, PV, PN)
    """

    # First Graph
    def first_fig():
        speeds = np.linspace(0.1, 300, 15)
        equipartition_temperature = []
        ideal_temperature = []
        pressure = []
        for i in speeds:
            mbs = simulations.MultiBallSimulation(b_radius=0.1, b_speed=i, nrings=3)
            mbs.run(2000, False, 0.01)
            equipartition_t = mbs.t_equipartition()
            ideal_t = mbs.t_ideal()

            # Y values
            equipartition_temperature.append(equipartition_t)
            ideal_temperature.append(ideal_t)
            # X value
            pressure.append(mbs.pressure())

            # Plotting
        plt.plot(equipartition_temperature, pressure, "-x", label="Measured Pressure")
        plt.plot(ideal_temperature, pressure, "-x", label="Ideal pressure")
        plt.xlabel("Temperature (K)")
        plt.ylabel("Pressure (Pa)")
        plt.legend()
        plt.title("Pressure against Equipartition Temperature and IGL Temperature")
        plt.savefig("fig13_PT", dpi=800)
        plt.show()

    # Second Graph
    def second_fig():
        container_radii = np.linspace(10, 20, 10)
        measured_pressure = []
        ideal_pressure = []
        container_volume = []
        for cont_radius in container_radii:
            mbs = simulations.MultiBallSimulation(b_radius=0.1, c_radius=cont_radius)
            container_volume.append(mbs.container().volume())
            mbs.run(2000, False, 0.01)
            # Y values
            ideal_pressure.append(mbs.p_ideal())
            measured_pressure.append(mbs.pressure())

        plt.plot(container_volume, measured_pressure, "-x", label="Measured Pressure")
        plt.plot(container_volume, ideal_pressure, "-x", label="Ideal Pressure")
        plt.xlabel(r"Container Volume ($m^3$)")
        plt.ylabel(r"Pressure (Pa)")
        plt.title(
            "The measured pressure and ideal pressure for different container volumes"
        )
        plt.legend()

        plt.savefig("fig13_PV", dpi=800)

    def third_fig():
        nballs = []
        ideal_pressure = []
        measured_pressure = []
        for i in range(1, 10):
            start = datetime.now()
            mbs = simulations.MultiBallSimulation(rmax=8, nrings=i, b_radius=0.1)
            mbs.run(0, False, 0.01)
            print(len(mbs.balls()))

        #     nballs.append(len(mbs.balls()))
        #     ideal_pressure.append(mbs.p_ideal())
        #     measured_pressure.append(mbs.pressure())
        #     end = datetime.now()
        #     print(f"Time taken for {len(mbs.balls())} balls: {end - start}s")

        # plt.plot(nballs, measured_pressure, "-x", label="Measured Pressure")
        # plt.plot(nballs, ideal_pressure, "-x", label="Ideal Pressure")
        # plt.xlabel("Number of Balls")
        # plt.ylabel("Pressure (Pa)")
        # plt.legend()
        # plt.title(
        #     "The measured pressure and ideal pressure for different numbers of balls"
        # )
        # plt.savefig("fig13_PN", dpi=800)

    first_fig()


def task14():
    """
    Task 14.

    In this function we shall be looking at the divergence of our simulation from the IGL. We shall
    quantify the ball radii dependence of this divergence by plotting the temperature ratio defined in
    the project brief.

    Returns:
        Figure: The temperature ratio figure.
    """
    ball_radii = np.linspace(0.01, 1, 15)
    temperature_ratio = []
    for radius in ball_radii:
        mbs = simulations.MultiBallSimulation(b_radius=radius)
        mbs.run(2000, False, 0.01)

        equipartition_temperature = mbs.t_equipartition()
        ideal_temperature = mbs.t_ideal()
        t_ratio = equipartition_temperature / ideal_temperature
        temperature_ratio.append(t_ratio)

    plt.plot(ball_radii, temperature_ratio, "-", label="Temperature Ratio")
    plt.plot(ball_radii, temperature_ratio, "x")
    plt.xlabel("Ball Radius (m)")
    plt.ylabel("Temperature Ratio")
    plt.title(
        "Equipartition Temperature and Ideal Temperature Ratio against Ball Radius"
    )
    plt.legend()
    plt.savefig("fig14", dpi=800)

    return


def task15():
    """
    Task 15.

    In this function we shall plot a histogram to investigate how the speeds of the balls evolve from the initial
    value. We shall then compare this to the Maxwell-Boltzmann distribution. Ensure that this function returns
    the created histogram.

    Returns:
        Figure: The speed histogram.
    """
    mbs = simulations.MultiBallSimulation(nrings=10, b_radius=0.1)
    mbs.run(3000, False, 0.01)

    speeds = mbs.speeds()
    max_speed = max(speeds) + 1
    kbt_ideal = sp.constants.Boltzmann * mbs.t_ideal()
    v = np.linspace(0, max_speed, 5000)
    f_v = physics.maxwell(v, kbt=kbt_ideal)

    plt.hist(
        speeds,
        bins=20,
        edgecolor="black",
        label="Normalized measured speed distribution",
        density=True,
    )
    plt.plot(v, f_v, label="Ideal Maxwell-Boltzmann Distribution")
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Probability")
    plt.title(
        "Comparison between the measured normalized speed distribution and the ideal Maxwell-Boltzmann distribution",
        fontsize=8,
    )
    plt.legend(fontsize=8)
    plt.savefig("fig15", dpi=800)
    plt.show()

    return


if __name__ == "__main__":

    # Run task 9 function
    task9()

    # Run task 10 function
    # task10()

    # Run task 11 function
    # task11()
    # FIG11_BALLCENTRE, FIG11_INTERBALL = task11()

    # Run task 12 function
    # task12()
    # FIG12_KE, FIG12_MOMX, FIG12_MOMY, FIG12_PT = task12()

    # Run task 13 function
    # task13()
    # FIG13_PT, FIG13_PV, FIG13_PN = task13()

    # Run task 14 function
    # task14()

    # Run task 15 function
    # task15()

    plt.show()
