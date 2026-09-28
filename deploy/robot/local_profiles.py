"""Station profiles added locally on top of the upstream PROFILES in config.py."""


def build_local_profiles(profile, default_init_q):
    return {
        "local_yam_config": profile(
            # (top, left wrist, right wrist) RealSense D405 serials.
            camera_serials=("427622273082", "427622272514", "427622271843"),
            # No GELLO leaders on this station; empty lets the driver auto-detect.
            leader_devices=("", ""),
            init_q=default_init_q,
            # (left lead, left follower, right lead, right follower) CAN USB serials.
            can_serials=("", "205434A258455017", "", "209037804546500A"),
        ),
    }
