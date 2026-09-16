"""Environment stage: the Gymnasium quadrotor racing environment.

Wraps an open quadrotor simulator (gym-pybullet-drones) as a Gymnasium env with a
timed waypoint/gate course. Penalizes collisions with floor, ceiling, and obstacles;
rewards minimal door-to-door (gate-to-gate) time.
"""
