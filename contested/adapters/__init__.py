"""Thin framework adapters over the shared canonical core (plan Section 3.12).

Both adapters are pure views over one :class:`~contested.core.CanonicalCore`; they
never duplicate dynamics. ``gym_vec`` exposes the Gymnasium vector API (TENURE's
own PPO, SB3); ``pettingzoo_par`` (later) exposes a ParallelEnv for BenchMARL.
"""
