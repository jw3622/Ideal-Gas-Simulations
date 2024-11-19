The program simulates ideal gas behaviour through treating individual particles as snooker balls with ideal gas assumptions:
  1. Collisions are elastic, conserving kinetic energy and momentum.
  2. Intermolecular forces are negligible except during collisions. (Considering no potential forces like Columb or Lennard-Jones potentials)
  3. Particles are initialized with random motion but the same velocity

File specifications: 

Balls.py : Initializes the thermalsnookers classes used in the simulation, giving them properties useful in the simulation
physics.py : Calculates the maxwell distribution probabilities for the configuration of thermalsnookers
Simulations.py : Sets up simulation scenarios and deals with visualization of the simulations 
Analysis : Performs analysis on temperature and total energy of the system, while validating conservation of momentum and energy, verifying that maxwell distribution holds for the ideal gas simulations
