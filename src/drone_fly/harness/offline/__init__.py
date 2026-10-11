"""Offline tooling: track/race-XML ground truth and estimator evaluation on recorded sessions.

This is the ONLY place Liftoff track/race XMLs are read. Runtime harness modules never import
this package (the model must perceive gates and obstacles from the screen, not from map files).
"""
