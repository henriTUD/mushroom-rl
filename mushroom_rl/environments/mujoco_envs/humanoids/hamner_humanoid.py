from pathlib import Path

import numpy as np
from dm_control import mjcf

from mushroom_rl.environments.mujoco_envs.humanoids.base_humanoid import BaseHumanoid
from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *

# optional imports
try:
    mujoco_viewer_available = True
    import mujoco_viewer
except ModuleNotFoundError:
    mujoco_viewer_available = False


class HamnerHumanoid(BaseHumanoid):
    """
    Mujoco simulation of full humanoid with muscle-actuated lower limb and torque-actuated upper body.

    """
    def __init__(self, obs_muscle_lens=False, obs_muscle_vels=False, obs_muscle_forces=False, use_brick_foots=False, disable_arms=False, tmp_dir_name=None,
                 xml_file_name='hamner_humanoid.xml', strict_reset_condition=True, **kwargs):
        """
        Constructor.

        """
        print("USING XML: " + xml_file_name)
        if use_brick_foots:
            assert tmp_dir_name is not None, "If you want to use brick foots or disable the arms, you have to specify a" \
                                             "directory name for the xml-files to be saved."
        xml_path = (Path(__file__).resolve().parent.parent / "data" / "hamner_humanoid" / xml_file_name).as_posix()
        self.xml_file_name = xml_file_name

        self.strict_reset_condition = strict_reset_condition

        self.arm_motors = [
            "shoulder_flex_r", "shoulder_add_r", "shoulder_rot_r",
            "elbow_flex_r", "pro_sup_r", "wrist_flex_r", "wrist_dev_r", "shoulder_flex_l", "shoulder_add_l",
            "shoulder_rot_l", "elbow_flex_l", "pro_sup_l", "wrist_flex_l", "wrist_dev_l"
        ]
        self.muscles = [
            "glut_med1_r", "glut_med2_r", "glut_med3_r", "glut_min1_r", "glut_min2_r", "glut_min3_r",
            "semimem_r", "semiten_r", "bifemlh_r", "bifemsh_r", "sar_r", "add_long_r", "add_brev_r",
            "add_mag1_r", "add_mag2_r", "add_mag3_r", "tfl_r", "pect_r", "grac_r", "glut_max1_r",
            "glut_max2_r", "glut_max3_r", "iliacus_r", "psoas_r", "quad_fem_r", "gem_r", "peri_r",
            "rect_fem_r", "vas_med_r", "vas_int_r", "vas_lat_r", "med_gas_r", "lat_gas_r",
            "soleus_r", "tib_post_r", "flex_dig_r", "flex_hal_r", "tib_ant_r", "per_brev_r",
            "per_long_r", "per_tert_r", "ext_dig_r", "ext_hal_r", "glut_med1_l", "glut_med2_l",
            "glut_med3_l", "glut_min1_l", "glut_min2_l", "glut_min3_l", "semimem_l", "semiten_l",
            "bifemlh_l", "bifemsh_l", "sar_l", "add_long_l", "add_brev_l", "add_mag1_l", "add_mag2_l",
            "add_mag3_l", "tfl_l", "pect_l", "grac_l", "glut_max1_l", "glut_max2_l", "glut_max3_l",
            "iliacus_l", "psoas_l", "quad_fem_l", "gem_l", "peri_l", "rect_fem_l", "vas_med_l",
            "vas_int_l", "vas_lat_l", "med_gas_l", "lat_gas_l", "soleus_l", "tib_post_l",
            "flex_dig_l", "flex_hal_l", "tib_ant_l", "per_brev_l", "per_long_l", "per_tert_l",
            "ext_dig_l", "ext_hal_l", "ercspn_r", "ercspn_l", "intobl_r", "intobl_l",
            "extobl_r", "extobl_l",
        ]

        action_spec = []
        action_spec.extend(self.arm_motors)
        action_spec.extend(self.muscles)

        observation_spec = [#------------- JOINT POS -------------
                            ("q_pelvis_tx", "pelvis_tx", ObservationType.JOINT_POS),
                            ("q_pelvis_tz", "pelvis_tz", ObservationType.JOINT_POS),
                            ("q_pelvis_ty", "pelvis_ty", ObservationType.JOINT_POS),
                            ("q_pelvis_tilt", "pelvis_tilt", ObservationType.JOINT_POS),
                            ("q_pelvis_list", "pelvis_list", ObservationType.JOINT_POS),
                            ("q_pelvis_rotation", "pelvis_rotation", ObservationType.JOINT_POS),
                            # --- lower limb right ---
                            ("q_hip_flexion_r", "hip_flexion_r", ObservationType.JOINT_POS),
                            ("q_hip_adduction_r", "hip_adduction_r", ObservationType.JOINT_POS),
                            ("q_hip_rotation_r", "hip_rotation_r", ObservationType.JOINT_POS),
                            #("q_knee_angle_r_translation2", "knee_angle_r_translation2", ObservationType.JOINT_POS),
                            #("q_knee_angle_r_translation1", "knee_angle_r_translation1", ObservationType.JOINT_POS),
                            ("q_knee_angle_r", "knee_angle_r", ObservationType.JOINT_POS),
                            #("q_knee_angle_r_rotation2", "knee_angle_r_rotation2", ObservationType.JOINT_POS),
                            #("q_knee_angle_r_rotation3", "knee_angle_r_rotation3", ObservationType.JOINT_POS),
                            ("q_ankle_angle_r", "ankle_angle_r", ObservationType.JOINT_POS),
                            ("q_subtalar_angle_r", "subtalar_angle_r", ObservationType.JOINT_POS),
                            ("q_mtp_angle_r", "mtp_angle_r", ObservationType.JOINT_POS),
                            #("q_knee_angle_r_beta_translation2", "knee_angle_r_beta_translation2", ObservationType.JOINT_POS),
                            #("q_knee_angle_r_beta_translation1", "knee_angle_r_beta_translation1", ObservationType.JOINT_POS),
                            #("q_knee_angle_r_beta_rotation1", "knee_angle_r_beta_rotation1", ObservationType.JOINT_POS),
                            # --- lower limb left ---
                            ("q_hip_flexion_l", "hip_flexion_l", ObservationType.JOINT_POS),
                            ("q_hip_adduction_l", "hip_adduction_l", ObservationType.JOINT_POS),
                            ("q_hip_rotation_l", "hip_rotation_l", ObservationType.JOINT_POS),
                            #("q_knee_angle_l_translation2", "knee_angle_l_translation2", ObservationType.JOINT_POS),
                            #("q_knee_angle_l_translation1", "knee_angle_l_translation1", ObservationType.JOINT_POS),
                            ("q_knee_angle_l", "knee_angle_l", ObservationType.JOINT_POS),
                            #("q_knee_angle_l_rotation2", "knee_angle_l_rotation2", ObservationType.JOINT_POS),
                            #("q_knee_angle_l_rotation3", "knee_angle_l_rotation3", ObservationType.JOINT_POS),
                            ("q_ankle_angle_l", "ankle_angle_l", ObservationType.JOINT_POS),
                            ("q_subtalar_angle_l", "subtalar_angle_l", ObservationType.JOINT_POS),
                            ("q_mtp_angle_l", "mtp_angle_l", ObservationType.JOINT_POS),
                            #("q_knee_angle_l_beta_translation2", "knee_angle_l_beta_translation2", ObservationType.JOINT_POS),
                            #("q_knee_angle_l_beta_translation1", "knee_angle_l_beta_translation1", ObservationType.JOINT_POS),
                            #("q_knee_angle_l_beta_rotation1", "knee_angle_l_beta_rotation1", ObservationType.JOINT_POS),
                            # --- lumbar ---
                            ("q_lumbar_extension", "lumbar_extension", ObservationType.JOINT_POS),
                            ("q_lumbar_bending", "lumbar_bending", ObservationType.JOINT_POS),
                            ("q_lumbar_rotation", "lumbar_rotation", ObservationType.JOINT_POS),
                            # q-- upper body right ---
                            ("q_arm_flex_r", "arm_flex_r", ObservationType.JOINT_POS),
                            ("q_arm_add_r", "arm_add_r", ObservationType.JOINT_POS),
                            ("q_arm_rot_r", "arm_rot_r", ObservationType.JOINT_POS),
                            ("q_elbow_flex_r", "elbow_flex_r", ObservationType.JOINT_POS),
                            ("q_pro_sup_r", "pro_sup_r", ObservationType.JOINT_POS),
                            ("q_wrist_flex_r", "wrist_flex_r", ObservationType.JOINT_POS),
                            ("q_wrist_dev_r", "wrist_dev_r", ObservationType.JOINT_POS),
                            # --- upper body left ---
                            ("q_arm_flex_l", "arm_flex_l", ObservationType.JOINT_POS),
                            ("q_arm_add_l", "arm_add_l", ObservationType.JOINT_POS),
                            ("q_arm_rot_l", "arm_rot_l", ObservationType.JOINT_POS),
                            ("q_elbow_flex_l", "elbow_flex_l", ObservationType.JOINT_POS),
                            ("q_pro_sup_l", "pro_sup_l", ObservationType.JOINT_POS),
                            ("q_wrist_flex_l", "wrist_flex_l", ObservationType.JOINT_POS),
                            ("q_wrist_dev_l", "wrist_dev_l", ObservationType.JOINT_POS),

                            # ------------- JOINT VEL -------------
                            ("dq_pelvis_tx", "pelvis_tx", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tz", "pelvis_tz", ObservationType.JOINT_VEL),
                            ("dq_pelvis_ty", "pelvis_ty", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tilt", "pelvis_tilt", ObservationType.JOINT_VEL),
                            ("dq_pelvis_list", "pelvis_list", ObservationType.JOINT_VEL),
                            ("dq_pelvis_rotation", "pelvis_rotation", ObservationType.JOINT_VEL),
                            # --- lower limb right ---
                            ("dq_hip_flexion_r", "hip_flexion_r", ObservationType.JOINT_VEL),
                            ("dq_hip_adduction_r", "hip_adduction_r", ObservationType.JOINT_VEL),
                            ("dq_hip_rotation_r", "hip_rotation_r", ObservationType.JOINT_VEL),
                            #("dq_knee_angle_r_translation2", "knee_angle_r_translation2", ObservationType.JOINT_VEL),
                            #("dq_knee_angle_r_translation1", "knee_angle_r_translation1", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_r", "knee_angle_r", ObservationType.JOINT_VEL),
                            #("dq_knee_angle_r_rotation2", "knee_angle_r_rotation2", ObservationType.JOINT_VEL),
                            #("dq_knee_angle_r_rotation3", "knee_angle_r_rotation3", ObservationType.JOINT_VEL),
                            ("dq_ankle_angle_r", "ankle_angle_r", ObservationType.JOINT_VEL),
                            ("dq_subtalar_angle_r", "subtalar_angle_r", ObservationType.JOINT_VEL),
                            ("dq_mtp_angle_r", "mtp_angle_r", ObservationType.JOINT_VEL),
                            #("dq_knee_angle_r_beta_translation2", "knee_angle_r_beta_translation2", ObservationType.JOINT_VEL),
                            #("dq_knee_angle_r_beta_translation1", "knee_angle_r_beta_translation1", ObservationType.JOINT_VEL),
                            #("dq_knee_angle_r_beta_rotation1", "knee_angle_r_beta_rotation1", ObservationType.JOINT_VEL),
                            # --- lower limb left ---
                            ("dq_hip_flexion_l", "hip_flexion_l", ObservationType.JOINT_VEL),
                            ("dq_hip_adduction_l", "hip_adduction_l", ObservationType.JOINT_VEL),
                            ("dq_hip_rotation_l", "hip_rotation_l", ObservationType.JOINT_VEL),
                            #("dq_knee_angle_l_translation2", "knee_angle_l_translation2", ObservationType.JOINT_VEL),
                            #("dq_knee_angle_l_translation1", "knee_angle_l_translation1", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l", "knee_angle_l", ObservationType.JOINT_VEL),
                            #("dq_knee_angle_l_rotation2", "knee_angle_l_rotation2", ObservationType.JOINT_VEL),
                            #("dq_knee_angle_l_rotation3", "knee_angle_l_rotation3", ObservationType.JOINT_VEL),
                            ("dq_ankle_angle_l", "ankle_angle_l", ObservationType.JOINT_VEL),
                            ("dq_subtalar_angle_l", "subtalar_angle_l", ObservationType.JOINT_VEL),
                            ("dq_mtp_angle_l", "mtp_angle_l", ObservationType.JOINT_VEL),
                            #("dq_knee_angle_l_beta_translation2", "knee_angle_l_beta_translation2", ObservationType.JOINT_VEL),
                            #("dq_knee_angle_l_beta_translation1", "knee_angle_l_beta_translation1", ObservationType.JOINT_VEL),
                            #("dq_knee_angle_l_beta_rotation1", "knee_angle_l_beta_rotation1", ObservationType.JOINT_VEL),
                            # --- lumbar ---
                            ("dq_lumbar_extension", "lumbar_extension", ObservationType.JOINT_VEL),
                            ("dq_lumbar_bending", "lumbar_bending", ObservationType.JOINT_VEL),
                            ("dq_lumbar_rotation", "lumbar_rotation", ObservationType.JOINT_VEL),
                            # --- upper body right ---
                            ("dq_arm_flex_r", "arm_flex_r", ObservationType.JOINT_VEL),
                            ("dq_arm_add_r", "arm_add_r", ObservationType.JOINT_VEL),
                            ("dq_arm_rot_r", "arm_rot_r", ObservationType.JOINT_VEL),
                            ("dq_elbow_flex_r", "elbow_flex_r", ObservationType.JOINT_VEL),
                            ("dq_pro_sup_r", "pro_sup_r", ObservationType.JOINT_VEL),
                            ("dq_wrist_flex_r", "wrist_flex_r", ObservationType.JOINT_VEL),
                            ("dq_wrist_dev_r", "wrist_dev_r", ObservationType.JOINT_VEL),
                            # --- upper body left ---
                            ("dq_arm_flex_l", "arm_flex_l", ObservationType.JOINT_VEL),
                            ("dq_arm_add_l", "arm_add_l", ObservationType.JOINT_VEL),
                            ("dq_arm_rot_l", "arm_rot_l", ObservationType.JOINT_VEL),
                            ("dq_elbow_flex_l", "elbow_flex_l", ObservationType.JOINT_VEL),
                            ("dq_pro_sup_l", "pro_sup_l", ObservationType.JOINT_VEL),
                            ("dq_wrist_flex_l", "wrist_flex_l", ObservationType.JOINT_VEL),
                            ("dq_wrist_dev_l", "wrist_dev_l", ObservationType.JOINT_VEL)]

        observation_spec.extend(self.build_muscle_observation_spec(obs_muscle_lens, obs_muscle_vels, obs_muscle_forces))

        collision_groups = [("floor", ["floor"]),
                            ("foot_r", ["r_foot"]),
                            ("front_foot_r", ["r_bofoot"]),
                            ("foot_l", ["l_foot"]),
                            ("front_foot_l", ["l_bofoot"])]

        self._use_brick_foots = use_brick_foots
        self._disable_arms = disable_arms
        joints_to_remove = []
        actuators_to_remove = []
        equ_constr_to_remove = []
        if use_brick_foots:
            joints_to_remove += ["subtalar_angle_l", "mtp_angle_l", "subtalar_angle_r", "mtp_angle_r"]
            equ_constr_to_remove += [j + "_constraint" for j in joints_to_remove]
            # ToDo: think about a smarter way to not include foot force twice for bricks
            collision_groups = [("floor", ["floor"]),
                                ("foot_r", ["foot_brick_r"]),
                                ("front_foot_r", ["foot_brick_r"]),
                                ("foot_l", ["foot_brick_l"]),
                                ("front_foot_l", ["foot_brick_l"])]

        if disable_arms:
            joints_to_remove +=["arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r", "pro_sup_r", "wrist_flex_r",
                                "wrist_dev_r", "arm_flex_l", "arm_add_l", "arm_rot_l", "elbow_flex_l", "pro_sup_l",
                                "wrist_flex_l", "wrist_dev_l"]
            actuators_to_remove += ["shoulder_flex_r", "shoulder_add_r", "shoulder_rot_r", "elbow_flex_r",
                                    "pro_sup_r", "wrist_flex_r", "wrist_dev_r", "shoulder_flex_l",
                                    "shoulder_add_l", "shoulder_rot_l", "elbow_flex_l", "pro_sup_l",
                                    "wrist_flex_l", "wrist_dev_l"]
            equ_constr_to_remove += ["wrist_flex_r_constraint", "wrist_dev_r_constraint",
                                    "wrist_flex_l_constraint", "wrist_dev_l_constraint"]

        if use_brick_foots or disable_arms:
            obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
            observation_spec = [elem for elem in observation_spec if elem[0] not in obs_to_remove]
            # ToDo: there are probabily some muscles that act on the foot, but these are not removed when using brick foots.
            action_spec = [ac for ac in action_spec if ac not in actuators_to_remove]
            xml_handle = mjcf.from_path(xml_path)
            xml_handle = self.delete_from_xml_handle(xml_handle, joints_to_remove,
                                                     actuators_to_remove, equ_constr_to_remove)
            if use_brick_foots:
                xml_handle = self.add_brick_foots_to_xml_handle(xml_handle)
            xml_path = self.save_xml_handle(xml_handle, tmp_dir_name)

        super().__init__(xml_path, action_spec, observation_spec, collision_groups, **kwargs)

    def build_muscle_observation_spec(self, use_muscle_lens, use_muscle_vels, use_muscle_forces):
        muscle_obs_spec = []

        if use_muscle_lens:
            muscle_len_obs = []
            for muscle in self.muscles:
                muscle_len_obs.append(('len_' + muscle, muscle, ObservationType.MUSCLE_LEN))
            muscle_obs_spec.extend(muscle_len_obs)
        if use_muscle_vels:
            muscle_vel_obs = []
            for muscle in self.muscles:
                muscle_vel_obs.append(('vel_' + muscle, muscle, ObservationType.MUSCLE_VEL))
            muscle_obs_spec.extend(muscle_vel_obs)
        if use_muscle_forces:
            muscle_force_obs = []
            for muscle in self.muscles:
                muscle_force_obs.append(('force_' + muscle, muscle, ObservationType.MUSCLE_FORCE))
            muscle_obs_spec.extend(muscle_force_obs)

        return muscle_obs_spec

    def delete_from_xml_handle(self, xml_handle, joints_to_remove, actuators_to_remove, equ_constraints):

        for j in joints_to_remove:
            j_handle = xml_handle.find("joint", j)
            j_handle.remove()
        for m in actuators_to_remove:
            m_handle = xml_handle.find("actuator", m)
            m_handle.remove()
        for e in equ_constraints:
            e_handle = xml_handle.find("equality", e)
            e_handle.remove()

        return xml_handle

    def add_brick_foots_to_xml_handle(self, xml_handle):

        # find foot and attach bricks
        toe_l = xml_handle.find("body", "toes_l")
        toe_l.add("geom", name="foot_brick_l", type="box", size=[0.112, 0.03, 0.05], pos=[-0.09, 0.019, 0.0],
                  rgba=[0.5, 0.5, 0.5, 0.5], euler=[0.0, 0.15, 0.0])
        toe_r = xml_handle.find("body", "toes_r")
        toe_r.add("geom", name="foot_brick_r", type="box", size=[0.112, 0.03, 0.05], pos=[-0.09, 0.019, 0.0],
                  rgba=[0.5, 0.5, 0.5, 0.5], euler=[0.0, -0.15, 0.0])

        # make true foot uncollidable
        foot_r = xml_handle.find("geom", "r_foot")
        bofoot_r = xml_handle.find("geom", "r_bofoot")
        foot_l = xml_handle.find("geom", "l_foot")
        bofoot_l = xml_handle.find("geom", "l_bofoot")
        foot_r.contype = 0
        foot_r.conaffinity = 0
        bofoot_r.contype = 0
        bofoot_r.conaffinity = 0
        foot_l.contype = 0
        foot_l.conaffinity = 0
        bofoot_l.contype = 0
        bofoot_l.conaffinity = 0

        return xml_handle

    def save_xml_handle(self, xml_handle, tmp_dir_name):

        # save new model and return new xml path
        new_model_dir_name = 'new_full_humanoid_with_bricks_model/' + tmp_dir_name + "/"
        cwd = Path.cwd()
        new_model_dir_path = Path.joinpath(cwd, new_model_dir_name)
        mjcf.export_with_assets(xml_handle, new_model_dir_path, self.xml_file_name)
        new_xml_path = Path.joinpath(new_model_dir_path, self.xml_file_name)

        return new_xml_path.as_posix()

    def has_fallen(self, state):
        pelvis_euler = state[1:4]

        if self._use_brick_foots:
            lumbar_euler = state[14:17]
        else:
            lumbar_euler = state[18:21]

        if self.strict_reset_condition:
            pelvis_condition = ((state[0] < -0.4) or (state[0] > 0.10)
                                or (pelvis_euler[0] < (-np.pi / 4.5)) or (pelvis_euler[0] > (np.pi / 12))
                                or (pelvis_euler[1] < -np.pi / 12) or (pelvis_euler[1] > np.pi / 8)
                                or (pelvis_euler[2] < (-np.pi / 10)) or (pelvis_euler[2] > (np.pi / 10))
                                )

            lumbar_condition = ((lumbar_euler[0] < (-np.pi / 6)) or (lumbar_euler[0] > (np.pi / 10))
                                or (lumbar_euler[1] < -np.pi / 10) or (lumbar_euler[1] > np.pi / 10)
                                or (lumbar_euler[2] < (-np.pi / 4.5)) or (lumbar_euler[2] > (np.pi / 4.5))
                                )
            return pelvis_condition or lumbar_condition
        else:
            pelvis_condition = ((state[0] < -0.35) or (state[0] > 0.10)
                                or (pelvis_euler[0] < (-np.pi / 4)) or (pelvis_euler[0] > (np.pi / 4))
                                or (pelvis_euler[1] < -np.pi / 4) or (pelvis_euler[1] > np.pi / 4)
                                or (pelvis_euler[2] < (-np.pi / 4)) or (pelvis_euler[2] > (np.pi / 4))
                                )

            lumbar_condition = ((lumbar_euler[0] < (-np.pi / 4)) or (lumbar_euler[0] > (np.pi / 4))
                                or (lumbar_euler[1] < -np.pi / 4) or (lumbar_euler[1] > np.pi / 4)
                                or (lumbar_euler[2] < (-np.pi / 4)) or (lumbar_euler[2] > (np.pi / 4))
                                )
            return lumbar_condition or pelvis_condition


if __name__ == '__main__':
    import time

    env = HamnerHumanoid(n_substeps=33, use_brick_foots=True, disable_arms=True, random_start=False,
                         obs_muscle_lens=True,
                         obs_muscle_vels=True,
                         obs_muscle_forces=False,
                         tmp_dir_name="test")
    print(env._n_intermediate_steps)
    print(env.get_obs_idx('q_pelvis_tilt'))
    print(env.get_obs_idx('q_hip_flexion_r'))
    print(env.get_obs_idx('dq_hip_flexion_r'))
    print(env.get_obs_idx('q_hip_flexion_l'))
    print(env.get_obs_idx('dq_hip_flexion_l'))

    action_dim = env.info.action_space.shape[0]
    state_dim = env.info.observation_space.shape[0]
    print("STATE DIM:")
    print(state_dim)

    env.reset()
    env.render()

    absorbing = False

    frequencies = 2*np.pi * np.ones(action_dim) * np.random.uniform(0, 10, action_dim)
    psi = np.zeros_like(frequencies)
    dt = 0.01

    ind = 0
    while True:
        ind += 1
        psi = psi + dt * frequencies
        action = np.sin(psi)
        action = np.random.normal(0.0, 0.5, (action_dim,)) # compare to normal gaussian noise
        nstate, _, absorbing, _ = env.step(action)

        env.render()

