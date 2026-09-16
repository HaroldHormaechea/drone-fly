"""Connectome stage: fetch MaleCNS connectivity from neuPrint and cache it locally.

Responsible for the only inbound external data. Reads are treated as read-only
reference data; the neuPrint auth token is supplied via the environment, never
committed. Bulk connectome data is cached under data/connectome/ and kept out of
version control for licensing reasons.
"""
