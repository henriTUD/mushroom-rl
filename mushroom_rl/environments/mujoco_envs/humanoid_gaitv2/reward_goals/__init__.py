from .reward import NoGoalReward, NoGoalRewardRandInit, MaxVelocityReward, \
    VelocityProfileReward, CompleteTrajectoryReward, ChangingVelocityTargetReward, CustomReward

from .trajectory import HumanoidTrajectory

from .velocity_profile import VelocityProfile, PeriodicVelocityProfile,\
    SinVelocityProfile, ConstantVelocityProfile, RandomConstantVelocityProfile,\
    SquareWaveVelocityProfile,  VelocityProfile3D