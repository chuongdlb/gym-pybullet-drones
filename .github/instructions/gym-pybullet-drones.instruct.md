Directory structure:
└── utiasdsl-gym-pybullet-drones/
    ├── README.md
    ├── CITATION.cff
    ├── LICENSE
    ├── pyproject.toml
    ├── gym_pybullet_drones/
    │   ├── __init__.py
    │   ├── assets/
    │   │   ├── architrave.urdf
    │   │   ├── beta-presets-bak.txt
    │   │   ├── box.urdf
    │   │   ├── cf2p.urdf
    │   │   ├── cf2x.urdf
    │   │   ├── clone_bfs.sh
    │   │   ├── ffmpeg_png2mp4.sh
    │   │   └── racer.urdf
    │   ├── control/
    │   │   ├── __init__.py
    │   │   ├── BaseControl.py
    │   │   ├── CTBRControl.py
    │   │   ├── DSLPIDControl.py
    │   │   └── MRAC.py
    │   ├── envs/
    │   │   ├── __init__.py
    │   │   ├── BaseRLAviary.py
    │   │   ├── BetaAviary.py
    │   │   ├── CFAviary.py
    │   │   ├── CtrlAviary.py
    │   │   ├── HoverAviary.py
    │   │   ├── MultiHoverAviary.py
    │   │   └── VelocityAviary.py
    │   ├── examples/
    │   │   ├── __init__.py
    │   │   ├── beta.py
    │   │   ├── cf.py
    │   │   ├── debug.py
    │   │   ├── downwash.py
    │   │   ├── learn.py
    │   │   ├── mrac.py
    │   │   ├── pid.py
    │   │   ├── pid_velocity.py
    │   │   └── play.py
    │   └── utils/
    │       ├── __init__.py
    │       ├── enums.py
    │       ├── Logger.py
    │       └── utils.py
    ├── tests/
    │   ├── test_build.py
    │   └── test_examples.py
    └── .github/
        ├── dependabot.yml
        └── workflows/
            └── test.yml

================================================
FILE: README.md
================================================
> [!TIP]
> For **symbolic dynamics and constraints**, also try our [`safe-control-gym`](https://github.com/utiasDSL/safe-control-gym)
>
> For **ROS2, PX4, ArduPilot, YOLO, and LiDAR**, also try our [`aerial-autonomy-stack`](https://github.com/JacopoPan/aerial-autonomy-stack)

# gym-pybullet-drones

This is a minimalist refactoring of the original `gym-pybullet-drones` repository, designed for compatibility with [`gymnasium`](https://github.com/Farama-Foundation/Gymnasium), [`stable-baselines3` 2.0](https://github.com/DLR-RM/stable-baselines3/pull/1327), and [`betaflight`](https://github.com/betaflight/betaflight)/[`crazyflie-firmware`](https://github.com/bitcraze/crazyflie-firmware/) SITL.

> **NOTE**: if you want to access the original codebase, presented at IROS in 2021, please `git checkout [paper|master]`

<img src="gym_pybullet_drones/assets/helix.gif" alt="formation flight" width="325"> <img src="gym_pybullet_drones/assets/helix.png" alt="control info" width="425">

## Installation

Tested on Intel x64/Ubuntu 22.04 and Apple Silicon/macOS 14.1.

```sh
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones/

conda create -n drones python=3.10
conda activate drones

pip3 install --upgrade pip
pip3 install -e . # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`

# check installed packages with `conda list`, deactivate with `conda deactivate`, remove with `conda remove -n drones --all`
```

## Use

### PID control examples

```sh
cd gym_pybullet_drones/examples/
python3 pid.py # position and velocity reference
python3 pid_velocity.py # desired velocity reference
```

### Downwash effect example

```sh
cd gym_pybullet_drones/examples/
python3 downwash.py
```

### Reinforcement learning examples (SB3's PPO)

```sh
cd gym_pybullet_drones/examples/
python learn.py # task: single drone hover at z == 1.0
python learn.py --multiagent true # task: 2-drone hover at z == 1.2 and 0.7
LATEST_MODEL=$(ls -t results | head -n 1) && python play.py --model_path "results/${LATEST_MODEL}/best_model.zip" # play and visualize the most recent learned policy after training
```

<img src="gym_pybullet_drones/assets/rl.gif" alt="rl example" width="375"> <img src="gym_pybullet_drones/assets/marl.gif" alt="marl example" width="375">

### Run all tests

```sh
# from the repo's top folder
cd gym-pybullet-drones/
pytest tests/
```

### utiasDSL `pycffirmware` Python Bindings example (multiplatform, single-drone)

Install [`pycffirmware`](https://github.com/utiasDSL/pycffirmware?tab=readme-ov-file#installation) for Ubuntu, macOS, or Windows

```sh
cd gym_pybullet_drones/examples/
python3 cff-dsl.py
```

### Betaflight SITL example (Ubuntu only)

```sh
git clone https://github.com/betaflight/betaflight 
cd betaflight/
git checkout cafe727 # `master` branch head at the time of writing (future release 4.5)
make arm_sdk_install # if needed, `apt install curl``
make TARGET=SITL # comment out line: https://github.com/betaflight/betaflight/blob/master/src/main/main.c#L52
cp ~/gym-pybullet-drones/gym_pybullet_drones/assets/eeprom.bin ~/betaflight/ # assuming both gym-pybullet-drones/ and betaflight/ were cloned in ~/
betaflight/obj/main/betaflight_SITL.elf
```

In another terminal, run the example

```sh
conda activate drones
cd gym_pybullet_drones/examples/
python3 beta.py --num_drones 1 # check the steps in the file's docstrings to use multiple drones
```

## Citation

If you wish, please cite our [IROS 2021 paper](https://arxiv.org/abs/2103.02142) ([and original codebase](https://github.com/utiasDSL/gym-pybullet-drones/tree/paper)) as

```bibtex
@INPROCEEDINGS{panerati2021learning,
      title={Learning to Fly---a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control}, 
      author={Jacopo Panerati and Hehui Zheng and SiQi Zhou and James Xu and Amanda Prorok and Angela P. Schoellig},
      booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
      year={2021},
      volume={},
      number={},
      pages={7512-7519},
      doi={10.1109/IROS51168.2021.9635857}
}
```

## References

- Erwin Coumans and Yunfei Bai (2023) [*PyBullet Quickstart Guide*](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit?tab=t.0#heading=h.2ye70wns7io3)
- Carlos Luis and Jeroome Le Ny (2016) [*Design of a Trajectory Tracking Controller for a Nanoquadcopter*](https://arxiv.org/pdf/1608.05786.pdf)
- Nathan Michael, Daniel Mellinger, Quentin Lindsey, Vijay Kumar (2010) [*The GRASP Multiple Micro-UAV Testbed*](https://ieeexplore.ieee.org/document/5569026)
- Benoit Landry (2014) [*Planning and Control for Quadrotor Flight through Cluttered Environments*](http://groups.csail.mit.edu/robotics-center/public_papers/Landry15)
- Julian Forster (2015) [*System Identification of the Crazyflie 2.0 Nano Quadrocopter*](https://www.research-collection.ethz.ch/handle/20.500.11850/214143)
- Antonin Raffin, Ashley Hill, Maximilian Ernestus, Adam Gleave, Anssi Kanervisto, and Noah Dormann (2019) [*Stable Baselines3*](https://github.com/DLR-RM/stable-baselines3)
- Guanya Shi, Xichen Shi, Michael O’Connell, Rose Yu, Kamyar Azizzadenesheli, Animashree Anandkumar, Yisong Yue, and Soon-Jo Chung (2019)
[*Neural Lander: Stable Drone Landing Control Using Learned Dynamics*](https://arxiv.org/pdf/1811.08027.pdf)
- C. Karen Liu and Dan Negrut (2020) [*The Role of Physics-Based Simulators in Robotics*](https://www.annualreviews.org/doi/pdf/10.1146/annurev-control-072220-093055)
- Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio, and Davide Scaramuzza (2020) [*Flightmare: A Flexible Quadrotor Simulator*](https://arxiv.org/pdf/2009.00563.pdf)

-----
> University of Toronto's [Dynamic Systems Lab](https://github.com/utiasDSL) / [Vector Institute](https://github.com/VectorInstitute) / University of Cambridge's [Prorok Lab](https://github.com/proroklab)

<!--
## WIP/Desired Contributions/PRs

- [ ] Multi-drone `crazyflie-firmware` SITL support
- [ ] Use SITL services with steppable simulation
- [ ] Add motor delay, advanced ESC modeling by implementing a buffer in `BaseAviary._dynamics()`
- [ ] Replace `rpy` with quaternions (and `ang_vel` with body rates) by editing `BaseAviary._updateAndStoreKinematicInformation()`, `BaseAviary._getDroneStateVector()`, and the `.computeObs()` methods of relevant subclasses

## Troubleshooting

- On Ubuntu, with an NVIDIA card, if you receive a "Failed to create and OpenGL context" message, launch `nvidia-settings` and under "PRIME Profiles" select "NVIDIA (Performance Mode)", reboot and try again.
-->



================================================
FILE: CITATION.cff
================================================
cff-version: 1.2.0
preferred-citation:
  type: article
  authors:
  - family-names: "Panerati"
    given-names: "Jacopo"
    orcid: "https://orcid.org/0000-0003-2994-5422"
  - family-names: "Zheng"
    given-names: "Hehui"
    orcid: "https://orcid.org/0000-0002-4977-0220"
  - family-names: "Zhou"
    given-names: "SiQi"
  - family-names: "Xu"
    given-names: "James"
  - family-names: "Prorok"
    given-names: "Amanda"
    orcid: "https://orcid.org/0000-0001-7313-5983"
  - family-names: "Schoellig"
    given-names: "Angela P."
    orcid: "https://orcid.org/0000-0003-4012-4668"
  doi: "10.0000/00000"
  journal: "2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)"
  month: 1
  start: 1 # First page number
  end: 8 # Last page number
  title: "Learning to Fly---a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control"
  issue: 1
  volume: 1
  year: 2021



================================================
FILE: LICENSE
================================================
MIT License

Copyright (c) 2020 Jacopo Panerati

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



================================================
FILE: pyproject.toml
================================================
[tool.poetry]
name = "gym-pybullet-drones"
version = "2.0.0"
description = "PyBullet Gym environments for single and multi-agent reinforcement learning of quadcopter control"
authors = ["Jacopo Panerati <jacopo.panerati@utoronto.ca>"]
license = "MIT"
readme = "README.md"
repository = "https://github.com/utiasDSL/gym-pybullet-drones"
packages = [
    { include = "gym_pybullet_drones" }
]

[tool.poetry.dependencies]
python = "^3.10"
numpy = "^2.2"
scipy = "^1.10"
transforms3d = "^0.4.1"
matplotlib = "^3.7"
pytest = ">=7.3,<10.0"
pybullet = "^3.2.5"
gymnasium = "~=1.2"
stable-baselines3 = "^2.0.0"
control = "^0.10.2"

[tool.poetry.dev-dependencies]

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"



================================================
FILE: gym_pybullet_drones/__init__.py
================================================
from gymnasium.envs.registration import register

register(
    id='ctrl-aviary-v0',
    entry_point='gym_pybullet_drones.envs:CtrlAviary',
)

register(
    id='velocity-aviary-v0',
    entry_point='gym_pybullet_drones.envs:VelocityAviary',
)

register(
    id='hover-aviary-v0',
    entry_point='gym_pybullet_drones.envs:HoverAviary',
)

register(
    id='multihover-aviary-v0',
    entry_point='gym_pybullet_drones.envs:MultiHoverAviary',
)



================================================
FILE: gym_pybullet_drones/assets/architrave.urdf
================================================
<?xml version="1.0" ?>

<robot name="architrave">

  <link name="base_link"> 

    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="0.3"/>
      <inertia ixx="0" ixy="0.0" ixz="0.0" iyy="0" iyz="0.0" izz="0"/>
    </inertial>

    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <box size=".8 .05 .05" />
      </geometry> 
      <material name="grey">
        <color rgba=".5 .5 .5 1"/>
      </material> 
    </visual>

    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <box size=".8 .05 .05" />
      </geometry>
    </collision>  

  </link>

</robot>




================================================
FILE: gym_pybullet_drones/assets/beta-presets-bak.txt
================================================
defaults nosave


# version
# Betaflight / SITL (SITL) 4.5.0 Jul  7 2023 / 10:53:28 (b54ae921f) MSP API: 1.46

# start the command batch
batch start

# reset configuration to default settings
defaults nosave

mcu_id 000000000000000100000002
signature 

# feature
feature -GPS
feature -TELEMETRY

# aux
aux 0 0 0 1300 1700 0 0

# master
set motor_pwm_protocol = PWM
set pid_process_denom = 16

profile 0

# profile 0
set pidsum_limit = 1000
set pidsum_limit_yaw = 1000
set p_pitch = 70
set i_pitch = 75
set d_pitch = 43
set f_pitch = 224
set p_roll = 58
set i_roll = 62
set d_roll = 42
set f_roll = 187
set p_yaw = 58
set i_yaw = 62
set f_yaw = 187
set d_min_roll = 42
set d_min_pitch = 43
set simplified_master_multiplier = 130
set simplified_i_gain = 60
set simplified_d_gain = 110
set simplified_dmax_gain = 0
set simplified_feedforward_gain = 120
set simplified_pitch_d_gain = 90
set simplified_pitch_pi_gain = 115

profile 1

profile 2

profile 3

# restore original profile selection
profile 0

rateprofile 0

# rateprofile 0
set rates_type = BETAFLIGHT
set roll_rc_rate = 180
set pitch_rc_rate = 180
set yaw_rc_rate = 180
set roll_srate = 0
set pitch_srate = 0
set yaw_srate = 0

rateprofile 1

rateprofile 2

rateprofile 3

# restore original rateprofile selection
rateprofile 0

# save configuration


================================================
FILE: gym_pybullet_drones/assets/box.urdf
================================================
<?xml version="1.0" ?>
<robot name="test.urdf">
  <link name="baseLink">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
       <mass value=".1"/>
       <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <box size=".05 .05 .1" />
      </geometry>
    </visual>
  </link>
</robot>


================================================
FILE: gym_pybullet_drones/assets/cf2p.urdf
================================================
<?xml version="1.0" ?>

<robot name="cf2">

  <properties arm="0.0397" kf="3.16e-10" km="7.94e-12" thrust2weight="2.25" max_speed_kmh="30" gnd_eff_coeff="11.36859" prop_radius="2.31348e-2" drag_coeff_xy="9.1785e-7" drag_coeff_z="10.311e-7" dw_coeff_1="2267.18" dw_coeff_2=".16" dw_coeff_3="-.11" />

  <link name="base_link"> 

    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="0.027"/>
      <inertia ixx="2.3951e-5" ixy="0.0" ixz="0.0" iyy="2.3951e-5" iyz="0.0" izz="3.2347e-5"/>
    </inertial>

    <!-- links>
      <carlos url="https://arxiv.org/pdf/1608.05786.pdf" />
      <julian url="https://www.research-collection.ethz.ch/handle/20.500.11850/214143" />
      <mit url="http://groups.csail.mit.edu/robotics-center/public_papers/Landry15.pdf" />
    </links -->

    <visual>
      <origin rpy="0 0 95" xyz="0 0 0"/>
      <geometry>
        <mesh filename="./cf2.dae" scale=" 1 1 1"/>
      </geometry> 
      <material name="grey">
        <color rgba=".5 .5 .5 1"/>
      </material> 
    </visual>

    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <cylinder radius=".06" length=".025"/>
      </geometry>
    </collision>  

  </link>

  <link name="prop0_link">
    <inertial>
      <origin rpy="0 0 0" xyz="0.0397 0 0"/>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="prop0_joint" type="fixed">
    <parent link="base_link"/>
    <child link="prop0_link"/>
  </joint>

  <link name="prop1_link">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0.0397 0"/>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="prop1_joint" type="fixed">
    <parent link="base_link"/>
    <child link="prop1_link"/>
  </joint>

  <link name="prop2_link">
    <inertial>
      <origin rpy="0 0 0" xyz="-0.0397 0 0"/>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="prop2_joint" type="fixed">
    <parent link="base_link"/>
    <child link="prop2_link"/>
  </joint>

  <link name="prop3_link">
    <inertial>
      <origin rpy="0 0 0" xyz="0 -0.0397 0"/>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="prop3_joint" type="fixed">
    <parent link="base_link"/>
    <child link="prop3_link"/>
  </joint>

  <link name="center_of_mass_link">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="center_of_mass_joint" type="fixed">
    <parent link="base_link"/>
    <child link="center_of_mass_link"/>
  </joint>

</robot>




================================================
FILE: gym_pybullet_drones/assets/cf2x.urdf
================================================
<?xml version="1.0" ?>

<robot name="cf2">

  <properties arm="0.0397" kf="3.16e-10" km="7.94e-12" thrust2weight="2.25" max_speed_kmh="30" gnd_eff_coeff="11.36859" prop_radius="2.31348e-2" drag_coeff_xy="9.1785e-7" drag_coeff_z="10.311e-7" dw_coeff_1="2267.18" dw_coeff_2=".16" dw_coeff_3="-.11" />

  <link name="base_link"> 

    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="0.027"/>
      <inertia ixx="1.4e-5" ixy="0.0" ixz="0.0" iyy="1.4e-5" iyz="0.0" izz="2.17e-5"/>
    </inertial>

    <!-- links>
      <carlos url="https://arxiv.org/pdf/1608.05786.pdf" />
      <julian url="https://www.research-collection.ethz.ch/handle/20.500.11850/214143" />
      <mit url="http://groups.csail.mit.edu/robotics-center/public_papers/Landry15.pdf" />
    </links -->

    <visual>
      <origin rpy="0 0 55" xyz="0 0 0"/>
      <geometry>
        <mesh filename="./cf2.dae" scale=" 1 1 1"/>
      </geometry> 
      <material name="grey">
        <color rgba=".5 .5 .5 1"/>
      </material> 
    </visual>

    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <cylinder radius=".06" length=".025"/>
      </geometry>
    </collision>  

  </link>

  <link name="prop0_link">
    <inertial>
      <origin rpy="0 0 0" xyz="0.028 -0.028 0"/>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="prop0_joint" type="fixed">
    <parent link="base_link"/>
    <child link="prop0_link"/>
  </joint>

  <link name="prop1_link">
    <inertial>
      <origin rpy="0 0 0" xyz="-0.028 -0.028 0"/>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="prop1_joint" type="fixed">
    <parent link="base_link"/>
    <child link="prop1_link"/>
  </joint>

  <link name="prop2_link">
    <inertial>
      <origin rpy="0 0 0" xyz="-0.028 0.028 0"/>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="prop2_joint" type="fixed">
    <parent link="base_link"/>
    <child link="prop2_link"/>
  </joint>

  <link name="prop3_link">
    <inertial>
      <origin rpy="0 0 0" xyz="0.028 0.028 0"/>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="prop3_joint" type="fixed">
    <parent link="base_link"/>
    <child link="prop3_link"/>
  </joint>

  <link name="center_of_mass_link">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="center_of_mass_joint" type="fixed">
    <parent link="base_link"/>
    <child link="center_of_mass_link"/>
  </joint>

</robot>




================================================
FILE: gym_pybullet_drones/assets/clone_bfs.sh
================================================
#!/bin/bash

# Clone, edit, build, configure, multiple Betaflight SITL executables

# Check for the correct number of command-line arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <desired_max_num_drones>"
    exit 1
fi

# Extract command-line arguments
desired_max_num_drones="$1"

# Create gitignored directory in gym-pybullet-donres
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
cd ../../
mkdir betaflight_sitl/
cd betaflight_sitl/

# Step 1: Clone and open betaflight's source (at the time of writing, branch `master`, future release 4.5)):
git clone https://github.com/betaflight/betaflight temp/


# Step 2: Comment out line `delayMicroseconds_real(50); // max rate 20kHz` 
# (https://github.com/betaflight/betaflight/blob/master/src/main/main.c#L52) 
# from Betaflight's `SIMULATOR_BUILD`
cd temp/

git checkout cafe727 #latest commit at the time of writing the gym-pybullet-drones Readme

sed -i "s/delayMicroseconds_real(50);/\/\/delayMicroseconds_real(50);/g" ./src/main/main.c

# Prepare
make arm_sdk_install 

cd ..

pattern1="PORT_PWM_RAW    9001"
pattern2="PORT_PWM        9002"
pattern3="PORT_STATE      9003"
pattern4="PORT_RC         9004"
# pattern5="BASE_PORT 5760"

for ((i = 0; i < desired_max_num_drones; i++)); do

    # Copy
    cp -r temp/ "bf${i}/"
    cd "bf${i}/"

    # Step 3: Change the UDP ports used by each Betaflight SITL instance
    replacement1="PORT_PWM_RAW    90${i}1"
    sed -i "s/$pattern1/$replacement1/g" ./src/main/target/SITL/sitl.c
    replacement2="PORT_PWM    90${i}2"
    sed -i "s/$pattern2/$replacement2/g" ./src/main/target/SITL/sitl.c
    replacement3="PORT_STATE    90${i}3"
    sed -i "s/$pattern3/$replacement3/g" ./src/main/target/SITL/sitl.c
    replacement4="PORT_RC    90${i}4"
    sed -i "s/$pattern4/$replacement4/g" ./src/main/target/SITL/sitl.c
    # replacement5="BASE_PORT 57${i}0"
    # sed -i "s/$pattern5/$replacement5/g" ./src/main/drivers/serial_tcp.c

    # Build
    make TARGET=SITL

    cd ..

done

for ((i = 0; i < desired_max_num_drones; i++)); do

    # Step 4: Copy over the configured `eeprom.bin` file from folder 
    cp "${SCRIPT_DIR}/eeprom.bin" "bf${i}/"

done



================================================
FILE: gym_pybullet_drones/assets/ffmpeg_png2mp4.sh
================================================
#!/bin/bash

# USE
# 1. copy the script in the directory where the framse are saved: $ cp ./ffmpeg_png2mp4.sh ./video-MM-DD-YYYY_HH.MM.SS
# 2. correct the values of FRAME_RATE and RESOLUTION, if necessary (remember not to add spaces before of after the = sign)
# 3. make the script executable, if needed: $ cd ./video-MM-DD-YYYY_HH.MM.SS; $ chmod +x ./ffmpeg_png2mp4.sh
# 4. run the script to create "video.mp4" in the same directory: $ ./ffmpeg_png2mp4.sh

FRAME_RATE=24
RESOLUTION=640x480

ffmpeg -r $FRAME_RATE -f image2 -s $RESOLUTION -i frame_%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p video.mp4


================================================
FILE: gym_pybullet_drones/assets/racer.urdf
================================================
<?xml version="1.0" ?>

<robot name="racer">

  <properties arm="0.109" kf="8.47e-9" km="2.13e-11" thrust2weight="4.17" max_speed_kmh="200" gnd_eff_coeff="11.36859" prop_radius="12.7e-2" drag_coeff_xy="9.1785e-7" drag_coeff_z="10.311e-7" dw_coeff_1="2267.18" dw_coeff_2=".16" dw_coeff_3="-.11" />

  <link name="base_link"> 

    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="0.830"/>
      <inertia ixx=".003113" ixy="0.0" ixz="0.0" iyy=".003113" iyz="0.0" izz=".003113"/>
    </inertial>

    <visual>
      <origin rpy="0 0 55" xyz="0 0 0"/>
      <geometry>
        <mesh filename="./cf2.dae" scale="3 3 3"/>
      </geometry> 
      <material name="grey">
        <color rgba=".5 .5 .5 1"/>
      </material> 
    </visual>

    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <cylinder radius=".06" length=".025"/>
      </geometry>
    </collision>  

  </link>

  <link name="prop0_link">
    <inertial>
      <origin rpy="0 0 0" xyz="0.0850 0.0675 0"/>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="prop0_joint" type="fixed">
    <parent link="base_link"/>
    <child link="prop0_link"/>
  </joint>

  <link name="prop1_link">
    <inertial>
      <origin rpy="0 0 0" xyz="-0.0850 0.0675 0"/>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="prop1_joint" type="fixed">
    <parent link="base_link"/>
    <child link="prop1_link"/>
  </joint>

  <link name="prop2_link">
    <inertial>
      <origin rpy="0 0 0" xyz="-0.085 -0.0675 0"/>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="prop2_joint" type="fixed">
    <parent link="base_link"/>
    <child link="prop2_link"/>
  </joint>

  <link name="prop3_link">
    <inertial>
      <origin rpy="0 0 0" xyz="0.085 -0.0675 0"/>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="prop3_joint" type="fixed">
    <parent link="base_link"/>
    <child link="prop3_link"/>
  </joint>

  <link name="center_of_mass_link">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="center_of_mass_joint" type="fixed">
    <parent link="base_link"/>
    <child link="center_of_mass_link"/>
  </joint>

</robot>


================================================
FILE: gym_pybullet_drones/control/__init__.py
================================================
[Empty file]


================================================
FILE: gym_pybullet_drones/control/BaseControl.py
================================================
import os
import numpy as np
import xml.etree.ElementTree as etxml
import pkg_resources

from gym_pybullet_drones.utils.enums import DroneModel

class BaseControl(object):
    """Base class for control.

    Implements `__init__()`, `reset(), and interface `computeControlFromState()`,
    the main method `computeControl()` should be implemented by its subclasses.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        #### Set general use constants #############################
        self.DRONE_MODEL = drone_model
        """DroneModel: The type of drone to control."""
        self.GRAVITY = g*self._getURDFParameter('m')
        """float: The gravitational force (M*g) acting on each drone."""
        self.KF = self._getURDFParameter('kf')
        """float: The coefficient converting RPMs into thrust."""
        self.KM = self._getURDFParameter('km')
        """float: The coefficient converting RPMs into torque."""
        self.reset()

    ################################################################################

    def reset(self):
        """Reset the control classes.

        A general use counter is set to zero.

        """
        self.control_counter = 0

    ################################################################################

    def computeControlFromState(self,
                                control_timestep,
                                state,
                                target_pos,
                                target_rpy=np.zeros(3),
                                target_vel=np.zeros(3),
                                target_rpy_rates=np.zeros(3)
                                ):
        """Interface method using `computeControl`.

        It can be used to compute a control action directly from the value of key "state"
        in the `obs` returned by a call to BaseAviary.step().

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        state : ndarray
            (20,)-shaped array of floats containing the current state of the drone.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        """
        return self.computeControl(control_timestep=control_timestep,
                                   cur_pos=state[0:3],
                                   cur_quat=state[3:7],
                                   cur_vel=state[10:13],
                                   cur_ang_vel=state[13:16],
                                   target_pos=target_pos,
                                   target_rpy=target_rpy,
                                   target_vel=target_vel,
                                   target_rpy_rates=target_rpy_rates
                                   )

    ################################################################################

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """Abstract method to compute the control action for a single drone.

        It must be implemented by each subclass of `BaseControl`.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        """
        raise NotImplementedError

################################################################################

    def setPIDCoefficients(self,
                           p_coeff_pos=None,
                           i_coeff_pos=None,
                           d_coeff_pos=None,
                           p_coeff_att=None,
                           i_coeff_att=None,
                           d_coeff_att=None
                           ):
        """Sets the coefficients of a PID controller.

        This method throws an error message and exist is the coefficients
        were not initialized (e.g. when the controller is not a PID one).

        Parameters
        ----------
        p_coeff_pos : ndarray, optional
            (3,1)-shaped array of floats containing the position control proportional coefficients.
        i_coeff_pos : ndarray, optional
            (3,1)-shaped array of floats containing the position control integral coefficients.
        d_coeff_pos : ndarray, optional
            (3,1)-shaped array of floats containing the position control derivative coefficients.
        p_coeff_att : ndarray, optional
            (3,1)-shaped array of floats containing the attitude control proportional coefficients.
        i_coeff_att : ndarray, optional
            (3,1)-shaped array of floats containing the attitude control integral coefficients.
        d_coeff_att : ndarray, optional
            (3,1)-shaped array of floats containing the attitude control derivative coefficients.

        """
        ATTR_LIST = ['P_COEFF_FOR', 'I_COEFF_FOR', 'D_COEFF_FOR', 'P_COEFF_TOR', 'I_COEFF_TOR', 'D_COEFF_TOR']
        if not all(hasattr(self, attr) for attr in ATTR_LIST):
            print("[ERROR] in BaseControl.setPIDCoefficients(), not all PID coefficients exist as attributes in the instantiated control class.")
            exit()
        else:
            self.P_COEFF_FOR = self.P_COEFF_FOR if p_coeff_pos is None else p_coeff_pos
            self.I_COEFF_FOR = self.I_COEFF_FOR if i_coeff_pos is None else i_coeff_pos
            self.D_COEFF_FOR = self.D_COEFF_FOR if d_coeff_pos is None else d_coeff_pos
            self.P_COEFF_TOR = self.P_COEFF_TOR if p_coeff_att is None else p_coeff_att
            self.I_COEFF_TOR = self.I_COEFF_TOR if i_coeff_att is None else i_coeff_att
            self.D_COEFF_TOR = self.D_COEFF_TOR if d_coeff_att is None else d_coeff_att

    ################################################################################
    
    def _getURDFParameter(self,
                          parameter_name: str
                          ):
        """Reads a parameter from a drone's URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter to read.

        Returns
        -------
        float
            The value of the parameter.

        """
        #### Get the XML tree of the drone model to control ########
        URDF = self.DRONE_MODEL.value + ".urdf"
        path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+URDF)
        URDF_TREE = etxml.parse(path).getroot()
        #### Find and return the desired parameter #################
        if parameter_name == 'm':
            return float(URDF_TREE[1][0][1].attrib['value'])
        elif parameter_name in ['ixx', 'iyy', 'izz']:
            return float(URDF_TREE[1][0][2].attrib[parameter_name])
        elif parameter_name in ['arm', 'thrust2weight', 'kf', 'km', 'max_speed_kmh', 'gnd_eff_coeff' 'prop_radius', \
                                'drag_coeff_xy', 'drag_coeff_z', 'dw_coeff_1', 'dw_coeff_2', 'dw_coeff_3']:
            return float(URDF_TREE[0].attrib[parameter_name])
        elif parameter_name in ['length', 'radius']:
            return float(URDF_TREE[1][2][1][0].attrib[parameter_name])
        elif parameter_name == 'collision_z_offset':
            COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
            return COLLISION_SHAPE_OFFSETS[2]



================================================
FILE: gym_pybullet_drones/control/CTBRControl.py
================================================
import os
import numpy as np
import xml.etree.ElementTree as etxml
import pkg_resources
import socket 
import struct

from transforms3d.quaternions import rotate_vector, qconjugate, mat2quat, qmult
from transforms3d.utils import normalized_vector

from gym_pybullet_drones.utils.enums import DroneModel

class CTBRControl(object):
    """Base class for control.

    Implements `__init__()`, `reset(), and interface `computeControlFromState()`,
    the main method `computeControl()` should be implemented by its subclasses.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        #### Set general use constants #############################
        self.DRONE_MODEL = drone_model
        """DroneModel: The type of drone to control."""
        self.GRAVITY = g*self._getURDFParameter('m')
        """float: The gravitational force (M*g) acting on each drone."""
        self.KF = self._getURDFParameter('kf')
        """float: The coefficient converting RPMs into thrust."""
        self.KM = self._getURDFParameter('km')
        """float: The coefficient converting RPMs into torque."""
        
        self.reset()

    ################################################################################

    def reset(self):
        """Reset the control classes.

        A general use counter is set to zero.

        """
        self.control_counter = 0

    ################################################################################

    def computeControlFromState(self,
                                control_timestep,
                                state,
                                target_pos,
                                target_rpy=np.zeros(3),
                                target_vel=np.zeros(3),
                                target_rpy_rates=np.zeros(3)
                                ):
        """Interface method using `computeControl`.

        It can be used to compute a control action directly from the value of key "state"
        in the `obs` returned by a call to BaseAviary.step().

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        state : ndarray
            (20,)-shaped array of floats containing the current state of the drone.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.
        """

        return self.computeControl(control_timestep=control_timestep,
                                   cur_pos=state[0:3],
                                   cur_quat=np.array([state[6], state[3], state[4], state[5]]),
                                   cur_vel=state[10:13],
                                   cur_ang_vel=state[13:16],
                                   target_pos=target_pos,
                                   target_rpy=target_rpy,
                                   target_vel=target_vel,
                                   target_rpy_rates=target_rpy_rates
                                   )

    ################################################################################

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """Abstract method to compute the control action for a single drone.

        It must be implemented by each subclass of `BaseControl`.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        """
        assert(cur_pos.shape == (3,)), f"cur_pos {cur_pos.shape}"
        assert(cur_quat.shape == (4,)), f"cur_quat {cur_quat.shape}"
        assert(cur_vel.shape == (3,)), f"cur_vel {cur_vel.shape}"
        assert(cur_ang_vel.shape == (3,)), f"cur_ang_vel {cur_ang_vel.shape}"
        assert(target_pos.shape == (3,)), f"target_pos {target_pos.shape}"
        assert(target_rpy.shape == (3,)), f"target_rpy {target_rpy.shape}"
        assert(target_vel.shape == (3,)), f"target_vel {target_vel.shape}"
        assert(target_rpy_rates.shape == (3,)), f"target_rpy_rates {target_rpy_rates.shape}"

        G = np.array([.0, .0, -9.8])
        K_P = np.array([3., 3., 8.])
        K_D = np.array([2.5, 2.5, 5.])
        K_RATES = np.array([5., 5., 1.])
        P = target_pos - cur_pos
        D = target_vel - cur_vel
        tar_acc = K_P * P + K_D * D - G
        norm_thrust = np.dot(tar_acc, rotate_vector([.0, .0, 1.], cur_quat))
        # Calculate target attitude
        z_body = normalized_vector(tar_acc)
        x_body = normalized_vector(np.cross(np.array([.0, 1., .0]), z_body))
        y_body = normalized_vector(np.cross(z_body, x_body))
        tar_att = mat2quat(np.vstack([x_body, y_body, z_body]).T)
        # Calculate body rates
        q_error = qmult(qconjugate(cur_quat), tar_att)
        body_rates = 2 * K_RATES * q_error[1:]
        if q_error[0] < 0:
            body_rates = -body_rates

        return norm_thrust, *body_rates

################################################################################

    def setPIDCoefficients(self,
                           p_coeff_pos=None,
                           i_coeff_pos=None,
                           d_coeff_pos=None,
                           p_coeff_att=None,
                           i_coeff_att=None,
                           d_coeff_att=None
                           ):
        """Sets the coefficients of a PID controller.

        This method throws an error message and exist is the coefficients
        were not initialized (e.g. when the controller is not a PID one).

        Parameters
        ----------
        p_coeff_pos : ndarray, optional
            (3,1)-shaped array of floats containing the position control proportional coefficients.
        i_coeff_pos : ndarray, optional
            (3,1)-shaped array of floats containing the position control integral coefficients.
        d_coeff_pos : ndarray, optional
            (3,1)-shaped array of floats containing the position control derivative coefficients.
        p_coeff_att : ndarray, optional
            (3,1)-shaped array of floats containing the attitude control proportional coefficients.
        i_coeff_att : ndarray, optional
            (3,1)-shaped array of floats containing the attitude control integral coefficients.
        d_coeff_att : ndarray, optional
            (3,1)-shaped array of floats containing the attitude control derivative coefficients.

        """
        ATTR_LIST = ['P_COEFF_FOR', 'I_COEFF_FOR', 'D_COEFF_FOR', 'P_COEFF_TOR', 'I_COEFF_TOR', 'D_COEFF_TOR']
        if not all(hasattr(self, attr) for attr in ATTR_LIST):
            print("[ERROR] in BaseControl.setPIDCoefficients(), not all PID coefficients exist as attributes in the instantiated control class.")
            exit()
        else:
            self.P_COEFF_FOR = self.P_COEFF_FOR if p_coeff_pos is None else p_coeff_pos
            self.I_COEFF_FOR = self.I_COEFF_FOR if i_coeff_pos is None else i_coeff_pos
            self.D_COEFF_FOR = self.D_COEFF_FOR if d_coeff_pos is None else d_coeff_pos
            self.P_COEFF_TOR = self.P_COEFF_TOR if p_coeff_att is None else p_coeff_att
            self.I_COEFF_TOR = self.I_COEFF_TOR if i_coeff_att is None else i_coeff_att
            self.D_COEFF_TOR = self.D_COEFF_TOR if d_coeff_att is None else d_coeff_att

    ################################################################################
    
    def _getURDFParameter(self,
                          parameter_name: str
                          ):
        """Reads a parameter from a drone's URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter to read.

        Returns
        -------
        float
            The value of the parameter.

        """
        #### Get the XML tree of the drone model to control ########
        URDF = self.DRONE_MODEL.value + ".urdf"
        path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+URDF)
        URDF_TREE = etxml.parse(path).getroot()
        #### Find and return the desired parameter #################
        if parameter_name == 'm':
            return float(URDF_TREE[1][0][1].attrib['value'])
        elif parameter_name in ['ixx', 'iyy', 'izz']:
            return float(URDF_TREE[1][0][2].attrib[parameter_name])
        elif parameter_name in ['arm', 'thrust2weight', 'kf', 'km', 'max_speed_kmh', 'gnd_eff_coeff' 'prop_radius', \
                                'drag_coeff_xy', 'drag_coeff_z', 'dw_coeff_1', 'dw_coeff_2', 'dw_coeff_3']:
            return float(URDF_TREE[0].attrib[parameter_name])
        elif parameter_name in ['length', 'radius']:
            return float(URDF_TREE[1][2][1][0].attrib[parameter_name])
        elif parameter_name == 'collision_z_offset':
            COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
            return COLLISION_SHAPE_OFFSETS[2]



================================================
FILE: gym_pybullet_drones/control/DSLPIDControl.py
================================================
import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class DSLPIDControl(BaseControl):
    """PID control class for Crazyflies.

    Based on work conducted at UTIAS' DSL. Contributors: SiQi Zhou, James Xu, 
    Tracy Du, Mario Vukosavljev, Calvin Ngan, and Jingyuan Hou.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()
        self.P_COEFF_FOR = np.array([.4, .4, 1.25])
        self.I_COEFF_FOR = np.array([.05, .05, .05])
        self.D_COEFF_FOR = np.array([.2, .2, .5])
        self.P_COEFF_TOR = np.array([70000., 70000., 60000.])
        self.I_COEFF_TOR = np.array([.0, .0, 500.])
        self.D_COEFF_TOR = np.array([20000., 20000., 12000.])
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([ 
                                    [-.5, -.5, -1],
                                    [-.5,  .5,  1],
                                    [.5, .5, -1],
                                    [.5, -.5,  1]
                                    ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                                    [0, -1,  -1],
                                    [+1, 0, 1],
                                    [0,  1,  -1],
                                    [-1, 0, 1]
                                    ])
        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = np.zeros(3)
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    ################################################################################
    
    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_dslPIDPositionControl()` and `_dslPIDAttitudeControl()`.
        Parameter `cur_ang_vel` is unused.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        self.control_counter += 1
        thrust, computed_target_rpy, pos_e = self._dslPIDPositionControl(control_timestep,
                                                                         cur_pos,
                                                                         cur_quat,
                                                                         cur_vel,
                                                                         target_pos,
                                                                         target_rpy,
                                                                         target_vel
                                                                         )
        rpm = self._dslPIDAttitudeControl(control_timestep,
                                          thrust,
                                          cur_quat,
                                          computed_target_rpy,
                                          target_rpy_rates
                                          )
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]
    
    ################################################################################

    def _dslPIDPositionControl(self,
                               control_timestep,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel
                               ):
        """DSL's CF2.x PID position control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray
            (3,1)-shaped array of floats containing the desired velocity.

        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.

        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        self.integral_pos_e = self.integral_pos_e + pos_e*control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)
        #### PID target thrust #####################################
        target_thrust = np.multiply(self.P_COEFF_FOR, pos_e) \
                        + np.multiply(self.I_COEFF_FOR, self.integral_pos_e) \
                        + np.multiply(self.D_COEFF_FOR, vel_e) + np.array([0, 0, self.GRAVITY])
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        #### Target rotation #######################################
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        if np.any(np.abs(target_euler) > math.pi):
            print("\n[ERROR] ctrl it", self.control_counter, "in Control._dslPIDPositionControl(), values outside range [-pi,pi]")
        return thrust, target_euler, pos_e
    
    ################################################################################

    def _dslPIDAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               target_euler,
                               target_rpy_rates
                               ):
        """DSL's CF2.x PID attitude control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        target_euler : ndarray
            (3,1)-shaped array of floats containing the computed target Euler angles.
        target_rpy_rates : ndarray
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w,x,y,z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()),cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy)/control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e*control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        #### PID target torques ####################################
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) \
                         + np.multiply(self.D_COEFF_TOR, rpy_rates_e) \
                         + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
    
    ################################################################################

    def _one23DInterface(self,
                         thrust
                         ):
        """Utility function interfacing 1, 2, or 3D thrust input use cases.

        Parameters
        ----------
        thrust : ndarray
            Array of floats of length 1, 2, or 4 containing a desired thrust input.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the PWM (not RPMs) to apply to each of the 4 motors.

        """
        DIM = len(np.array(thrust))
        pwm = np.clip((np.sqrt(np.array(thrust)/(self.KF*(4/DIM)))-self.PWM2RPM_CONST)/self.PWM2RPM_SCALE, self.MIN_PWM, self.MAX_PWM)
        if DIM in [1, 4]:
            return np.repeat(pwm, 4/DIM)
        elif DIM==2:
            return np.hstack([pwm, np.flip(pwm)])
        else:
            print("[ERROR] in DSLPIDControl._one23DInterface()")
            exit()



================================================
FILE: gym_pybullet_drones/control/MRAC.py
================================================
import math
import numpy as np
import pybullet as p
import control as ct
from scipy.spatial.transform import Rotation
from scipy.linalg import solve_lyapunov

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel


class MRAC(BaseControl):
    """Model Reference Adaptive Controller class for Crazyflies.

        Based on the implementation of https://github.com/caoty777/Quadcoptor-Adaptive-Flight-Control
    """

    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL not in [DroneModel.CF2X, DroneModel.CF2P, DroneModel.RACE]:
            print("[ERROR] MRAC requires DroneModel.CF2X or DroneModel.CF2P or DroneModel.RACE")
            exit()
        self.Ixx = self._getURDFParameter("ixx")
        self.Iyy = self._getURDFParameter("iyy")
        self.Izz = self._getURDFParameter("izz")
        self.J = np.diag([self.Ixx, self.Iyy, self.Izz])
        self.mass = self._getURDFParameter("m")
        self.l = self._getURDFParameter("arm")
        self.g = g
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        self.Ka = self.KF
        self.Km = self.KM

        if self.DRONE_MODEL == DroneModel.CF2X or self.DRONE_MODEL == DroneModel.RACE:
            self.MIXER_MATRIX = np.array([ 
                                    [-.5, -.5, -1],
                                    [-.5,  .5,  1],
                                    [.5, .5, -1],
                                    [.5, -.5,  1]
                                    ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                                    [0, -1,  -1],
                                    [+1, 0, 1],
                                    [0,  1,  -1],
                                    [-1, 0, 1]
                                    ])

        self.Kx, self.Kr = self._compute_K()
        self.Xm = np.zeros((12))
        self.reset()

    def _compute_K(self, psi=0):
        """x = x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r
            u = [w1^2, w2^2, w3^2, w4^2] or [thrust, tx, ty, tz]
        """
        g = self.g
        m = self.mass
        Ixx = self.Ixx
        Iyy = self.Iyy
        Izz = self.Izz
        l = self.l
        Ka = self.Ka
        Km = self.Km

        a_sub = np.array([[0, 0, 0, g*np.sin(psi), g*np.cos(psi), 0],
                          [0, 0, 0, -g*np.cos(psi), g*np.sin(psi), 0]])
        a_sub = np.vstack((a_sub, np.zeros((4, 6))))
        
        A = np.block([[np.zeros((6,6)), np.eye(6)],
                      [a_sub, np.zeros((6,6))]])
        
        b_sub = np.array([[1/m, 0, 0, 0],
                          [0, 1/Ixx, 0, 0],
                          [0, 0, 1/Iyy, 0],
                          [0, 0, 0, 1/Izz]])    
        # b_sub = np.array([[Ka/m, Ka/m, Ka/m, Ka/m],
        #             [0, -Ka*l/Ixx, 0, Ka*l/Ixx],
        #             [Ka*l/Iyy, 0, -Ka*l/Iyy, 0],
        #             [Km/Izz, -Km/Izz, Km/Izz, -Km/Izz]]) # For direct rpm
        
        B = np.vstack((np.zeros((8, 4)), b_sub))
        Q = np.eye(12)*600
        # Q = np.diag((700, 700, 700, 500, 500, 500, 500, 500, 500, 500, 500, 500))
        # R = np.eye(4)*10
        # K, self.P, _ = ct.lqr(A, B, Q, R)

        desired_poles = -np.linspace(1, 12, 12)
        K = ct.place(A, B, desired_poles)
        self.Kr_ref_gain = np.linalg.pinv(B) @ (A - B @ K)

        self.Am = A - B@K
        self.Bm = np.copy(B)
        self.P = solve_lyapunov(self.Am.T, -Q)

        self.Gamma_x = np.eye(12) * 5e-3
        self.Gamma_r = np.eye(4) * 5e-3

        Kx = -K.T
        Kr = np.eye(4) 
        return Kx, Kr

    def reset(self):
        super().reset()

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)):
        
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        cur_ang_vel = Rotation.from_euler('XYZ', cur_rpy).inv().apply(cur_ang_vel) # Convert angular velocity to body frame
        
        if self.control_counter == 0:
            self.Xm = np.hstack((cur_pos, cur_rpy, cur_vel, cur_ang_vel)).reshape(12, 1)
        self.control_counter += 1

        r = np.hstack((target_pos, target_rpy, target_vel, target_rpy_rates)).reshape(12, 1)
        rt = -self.Kr_ref_gain @ r

        X_actual = np.hstack((cur_pos, cur_rpy, cur_vel, cur_ang_vel)).reshape(12, 1)
        u = self.Kx.T @ X_actual + self.Kr.T @ rt
        e = X_actual - self.Xm # TODO plot X_actual and Xm
        Kx_dot = -self.Gamma_x @ X_actual @ e.T @ self.P @ self.Bm
        Kr_dot = -self.Gamma_r @ rt @ e.T @ self.P @ self.Bm

        self.Kx += Kx_dot * control_timestep
        self.Kr += Kr_dot * control_timestep

        thrust, tx, ty, tz = u.squeeze()
        thrust = np.maximum(0, thrust)
        target_torques = np.hstack((tx, ty, tz))
        target_torques = np.clip(target_torques, -3200, 3200)

        thrust = (math.sqrt(thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
        
        pos_e = target_pos - cur_pos
        rpy_e = target_rpy - cur_rpy

        Xm_dot = self.Am @ self.Xm + self.Bm @ rt
        self.Xm += Xm_dot*control_timestep

        return rpm, pos_e, rpy_e



================================================
FILE: gym_pybullet_drones/envs/__init__.py
================================================
from gym_pybullet_drones.envs.BetaAviary import BetaAviary
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary



================================================
FILE: gym_pybullet_drones/envs/BaseRLAviary.py
================================================
import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class BaseRLAviary(BaseAviary):
    """Base single and multi-agent environment class for reinforcement learning."""
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a generic single and multi-agent RL environment.

        Attributes `vision_attributes` and `dynamics_attributes` are selected
        based on the choice of `obs` and `act`; `obstacles` is set to True 
        and overridden with landmarks for vision applications; 
        `user_debug_gui` is set to False for performance.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """
        #### Create a buffer for the last .5 sec of actions ########
        self.ACTION_BUFFER_SIZE = int(ctrl_freq//2)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        ####
        vision_attributes = True if obs == ObservationType.RGB else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
            else:
                print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obstacles=True, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=vision_attributes,
                         )
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            p.loadURDF("block.urdf",
                       [1, 0, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("cube_small.urdf",
                       [0, 1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("duck_vhacd.urdf",
                       [-1, 0, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("teddy_vhacd.urdf",
                       [0, -1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
        else:
            pass

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.

        """
        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
            size = 4
        elif self.ACT_TYPE==ActionType.PID:
            size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            size = 1
        else:
            print("[ERROR] in BaseRLAviary._actionSpace()")
            exit()
        act_lower_bound = np.array([-1*np.ones(size) for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
        #
        for i in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))
        #
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES,4))
        for k in range(action.shape[0]):
            target = action[k, :]
            if self.ACT_TYPE == ActionType.RPM:
                rpm[k,:] = np.array(self.HOVER_RPM * (1+0.05*target))
            elif self.ACT_TYPE == ActionType.PID:
                state = self._getDroneStateVector(k)
                next_pos = self._calculateNextStep(
                    current_position=state[0:3],
                    destination=target,
                    step_size=1,
                    )
                rpm_k, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=next_pos
                                                        )
                rpm[k,:] = rpm_k
            elif self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(k)
                if np.linalg.norm(target[0:3]) != 0:
                    v_unit_vector = target[0:3] / np.linalg.norm(target[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3], # same as the current position
                                                        target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                        target_vel=self.SPEED_LIMIT * np.abs(target[3]) * v_unit_vector # target the desired velocity vector
                                                        )
                rpm[k,:] = temp
            elif self.ACT_TYPE == ActionType.ONE_D_RPM:
                rpm[k,:] = np.repeat(self.HOVER_RPM * (1+0.05*target), 4)
            elif self.ACT_TYPE == ActionType.ONE_D_PID:
                state = self._getDroneStateVector(k)
                res, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3]+0.1*np.array([0,0,target[0]])
                                                        )
                rpm[k,:] = res
            else:
                print("[ERROR] in BaseRLAviary._preprocessAction()")
                exit()
        return rpm

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ
            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array([[lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo] for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi] for i in range(self.NUM_DRONES)])
            #### Add action buffer to observation space ################
            act_lo = -1
            act_hi = +1
            for i in range(self.ACTION_BUFFER_SIZE):
                if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE==ActionType.PID:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi] for i in range(self.NUM_DRONES)])])
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")
    
    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i,
                                                                                 segmentation=False
                                                                                 )
                    #### Printing observation to PNG frames example ############
                    if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB,
                                          img_input=self.rgb[i],
                                          path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                                          frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                          )
            return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            obs_12 = np.zeros((self.NUM_DRONES,12))
            for i in range(self.NUM_DRONES):
                #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                obs = self._getDroneStateVector(i)
                obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
            ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
            #### Add action buffer to observation #######################
            for i in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
            return ret
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._computeObs()")



================================================
FILE: gym_pybullet_drones/envs/BetaAviary.py
================================================
import numpy as np
from gymnasium import spaces
import socket
import struct 
import os
import subprocess
import time

from transforms3d.quaternions import rotate_vector, qconjugate

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

BASE_PORT_PWM=9002 # In
BASE_PORT_STATE=9003 # Out
BASE_PORT_RC=9004 # Out

class BetaAviary(BaseAviary):
    """Multi-drone environment class for use of BetaFlight controller."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results',
                 udp_ip="127.0.0.1"
                 ):
        """Initialization of an aviary environment for use of BetaFlight controller.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        udp_ip : base ip for betaflight controller emulator 

        """
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder
                         )
        
        # Spawn SITL Betaflight instances (must have been created with assets/clone_bfs/sh first)
        for i in range(num_drones):
            FOLDER = os.path.dirname(os.path.abspath(__file__))+'/../../betaflight_sitl/bf'+str(i)+'/'
            cmd = f"gnome-terminal -- bash -c 'cd {FOLDER} && ./obj/main/betaflight_SITL.elf; exec bash'"
            subprocess.Popen(cmd, shell=True)
        time.sleep(2)
        
        # Initialize connection to BetaFlight Controller 
        self.UDP_IP = udp_ip
        self.ARM_TIME = 1
        self.TRAJ_TIME = 1.5

        self.sock = []
        self.sock_pwm = []
        for i in range(self.NUM_DRONES):
            self.sock.append(socket.socket(socket.AF_INET,    # Internet
                                socket.SOCK_DGRAM)) # UDP
            self.sock_pwm.append(socket.socket(socket.AF_INET, # Internet
                                socket.SOCK_DGRAM))  # UDP
            self.sock_pwm[i].bind((self.UDP_IP, BASE_PORT_PWM + 10 * (i))) # drone0 will be on 9002, drone1 on 9012, etc.
            self.sock_pwm[i].settimeout(0.0)

        self.beta_action = np.zeros((self.NUM_DRONES, 4))
    
    ################################################################################

    def step(self, action, i):
        obs, reward, terminated, truncated, info = super().step(self.beta_action)

        t = i/self.CTRL_FREQ

        for j in range(self.NUM_DRONES): #TODO: add multi-drone support

            #### State message to Betaflight ###########################
            o = obs[j,:] # p, q, euler, v, w, rpm (all in world frame)
            p = o[:3]
            q = np.array([o[6], o[3], o[4], o[5]]) # w, x, y, z
            v = o[10:13]
            w = o[13:16] # world frame
            w_body = rotate_vector(w, qconjugate(q)) # local frame
            
            fdm_packet = struct.pack(
                '@dddddddddddddddddd', # t, w, a, q, v, p, pressure
                t, # double timestamp in seconds
                # minus signs due to ENU to NED conversion
                w_body[0], -w_body[1], -w_body[2], # double imu_angular_velocity_rpy[3]
                0, 0, 0, # double imu_linear_acceleration_xyz[3]
                1., .0, .0, 0., # double imu_orientation_quat[4] w, x, y, z
                0, 0, 0, # double velocity_xyz[3]
                0, 0, 0, # double position_xyz[3]
                1.0 # double pressure;
            )
            self.sock[j].sendto(fdm_packet, (self.UDP_IP, BASE_PORT_STATE + 10 * (j))) # drone0 will be on 9003, drone1 on 9013, etc.

            # Action is (thro, roll, pitch, yaw) 
            thro = 1000 # Positive up
            yaw = 1500 # Positive CCW
            pitch = 1500 # Positive forward in x
            roll = 1500 # Positive right/forward in y

            if t > self.TRAJ_TIME: 
                thro, roll, pitch, yaw = self.ctbr2beta(*action[j,:])

            #### RC message to Betaflight ##############################
            aux1 = 1000 if t < self.ARM_TIME else 1500 # Arm
            rc_packet = struct.pack(
                '@dHHHHHHHHHHHHHHHH', 
                t, # datetime.now().timestamp(), # double timestamp; // in seconds
                round(roll), round(pitch), round(thro), round(yaw),              # roll, pitch, throttle, yaw
                aux1, 1000, 1000, 1000,              # aux 1, ..
                1000, 1000, 1000, 1000, 
                1000, 1000, 1000, 1000
            )
            # print("rc", struct.unpack('@dHHHHHHHHHHHHHHHH', rc_packet))
            self.sock[j].sendto(rc_packet, (self.UDP_IP, BASE_PORT_RC + 10 * (j))) # drone0 will be on 9004, drone1 on 9014, etc.

            #### PWMs message from Betaflight ##########################
            try:
                data, addr = self.sock_pwm[j].recvfrom(16) # buffer size is 100 bytes (servo_packet size 16)
            except socket.error as msg:
                _action = self.beta_action[j,:]
                pass
            else:
                # print("received message: ", data)
                _action = np.array(struct.unpack('@ffff', data)).reshape((1,4))
            self.beta_action[j,:] = _action

        return obs, reward, terminated, truncated, info

    ################################################################################
    
    def ctbr2beta(self, thrust, roll, pitch, yaw):
        MIN_CHANNEL = 1000
        MAX_CHANNEL = 2000
        MAX_RATE = 360
        MAX_THRUST = 40.9
        mid = (MAX_CHANNEL + MIN_CHANNEL) / 2
        d = (MAX_CHANNEL - MIN_CHANNEL) / 2
        thrust = thrust / MAX_THRUST * d * 2 + MIN_CHANNEL
        rates = np.array([roll, pitch, -yaw])
        rates = rates / np.pi * 180 / MAX_RATE * d + mid
        thrust = np.clip(thrust, MIN_CHANNEL, MAX_CHANNEL)
        rates = np.clip(rates, MIN_CHANNEL, MAX_CHANNEL)
        return thrust, *rates

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            An ndarray of shape (NUM_DRONES, 4) for the commanded CTBR to Betaflight SITL.

        """
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([[0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        spaces.Box
            The observation space, i.e., an ndarray of shape (NUM_DRONES, 20).

        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array([[-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of the state, see the implementation of `_getDroneStateVector()`.

        Returns
        -------
        ndarray
            An ndarray of shape (NUM_DRONES, 20) with the state of each drone.

        """
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Clips and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : ndarray
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        remapped_input = np.array([[
                                    action[i][2], # 0
                                    action[i][1], # 1
                                    action[i][3], # 2
                                    action[i][0]  # 3
                                ] for i in range(self.NUM_DRONES)]) # Betaflight SITL motor mapping

        ret = np.array(np.sqrt(self.MAX_THRUST / 4 / self.KF * remapped_input))
        assert(ret.shape == (self.NUM_DRONES, 4)), "Error in preprocess action"
        return ret

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        return -1

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False
    
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years



================================================
FILE: gym_pybullet_drones/envs/CFAviary.py
================================================
import numpy as np
from gymnasium import spaces
import socket
import math 

from scipy.spatial.transform import Rotation as R

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
try:
    import pycffirmware as firm 
except ImportError:
    raise "PyCFFirmware required for CF Aviary. Please install it from https://github.com/utiasDSL/pycffirmware or use a different aviary class."


class CFAviary(BaseAviary):
    """Multi-drone environment class for use of BetaFlight controller."""
    ACTION_DELAY = 0 # how many firmware loops run between the controller commanding an action and the drone motors responding to it
    SENSOR_DELAY = 0 # how many firmware loops run between experiencing a motion and the sensors registering it
    STATE_DELAY = 0 # not yet supported, keep 0
    CONTROLLER = 'mellinger' # specifies controller type 

    # Configurations to match firmware. Not recommended to change
    GYRO_LPF_CUTOFF_FREQ = 80
    ACCEL_LPF_CUTOFF_FREQ = 30
    QUAD_FORMATION_X = True
    MOTOR_SET_ENABLE = True

    RAD_TO_DEG = 180 / math.pi

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 500,
                 ctrl_freq: int = 25,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results',
                 verbose=False,
                 ):
        """Initialization of an aviary environment for use of BetaFlight controller.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        udp_ip : base ip for betaflight controller emulator 

        """
        firmware_freq = 500 if self.CONTROLLER == "mellinger" else 1000
        assert (pyb_freq % firmware_freq == 0), f"pyb_freq ({pyb_freq}) must be a multiple of firmware_freq ({firmware_freq}) for CFAviary."
        if num_drones != 1: 
            raise NotImplementedError("Multi-agent support for CF Aviary is not yet implemented.")
        
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=firmware_freq, # ctrl_freq in this variable (self.CTRL_FREQ) corresponds to ctrl rate from aviary, in this case, firmware_freq
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder
                         )
        
        # Initialize connection to BetaFlight Controller 
        self.firmware_freq = firmware_freq
        self.ctrl_freq = ctrl_freq

        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        self.verbose = verbose

        self._initalize_cffirmware()
    
    def _initalize_cffirmware(self):
        """Resets the firmware_wrapper object.

        Todo:
            * Add support for state estimation 
        """
        self.states = []
        self.takeoff_sent = False

        # Initialize history  
        self.action_history = [[0, 0, 0, 0] for _ in range(self.ACTION_DELAY)]
        self.sensor_history = [[[0, 0, 0], [0, 0, 0]] for _ in range(self.SENSOR_DELAY)]
        self.state_history = [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]] for _ in range(self.STATE_DELAY)]

        # Initialize gyro lpf 
        self.acclpf = [firm.lpf2pData() for _ in range(3)]
        self.gyrolpf = [firm.lpf2pData() for _ in range(3)]
        for i in range(3):
            firm.lpf2pInit(self.acclpf[i], self.firmware_freq, self.GYRO_LPF_CUTOFF_FREQ)
            firm.lpf2pInit(self.gyrolpf[i], self.firmware_freq, self.ACCEL_LPF_CUTOFF_FREQ)
        
        # Initialize state objects 
        self.control = firm.control_t()
        self.setpoint = firm.setpoint_t()
        self.sensorData = firm.sensorData_t()
        self.state = firm.state_t()
        self.tick = 0
        self.pwms = [0, 0, 0, 0]
        self.action = np.array([[0, 0, 0, 0]])
        self.command_queue = []

        self.tumble_counter = 0
        self.prev_vel = np.array([0, 0, 0])
        self.prev_rpy = np.array([0, 0, 0])
        self.prev_time_s = None
        self.last_pos_pid_call = 0
        self.last_att_pid_call = 0
        
        # Initialize state flags 
        self._error = False
        self.sensorData_set = False
        self.state_set = False
        self.full_state_cmd_override = True # When true, high level commander is not called  

        # Initialize controller
        if self.CONTROLLER == 'pid':
            firm.controllerPidInit()
            print('PID controller init test:', firm.controllerPidTest())
        elif self.CONTROLLER == 'mellinger':
            firm.controllerMellingerInit()
            assert(self.firmware_freq == 500), "Mellinger controller requires a firmware frequency of 500Hz."
            print('Mellinger controller init test:', firm.controllerMellingerTest())
        
        # Reset environment 
        init_obs, init_info = super().reset()
        init_pos=np.array([init_obs[0][0], init_obs[0][1], init_obs[0][2]]) # global coord, m
        init_vel=np.array([init_obs[0][10], init_obs[0][11], init_obs[0][12]]) # global coord, m/s
        init_rpy = np.array([init_obs[0][7], init_obs[0][8], init_obs[0][9]]) # body coord, rad 
        if self.NUM_DRONES > 1: 
            raise NotImplementedError("Firmware controller wrapper does not support multiple drones.")

        # Initilaize high level commander 
        firm.crtpCommanderHighLevelInit()
        self._update_state(0, init_pos, init_vel, np.array([0.0, 0.0, 1.0]), init_rpy * self.RAD_TO_DEG)
        self._update_initial_state(init_obs[0])
        firm.crtpCommanderHighLevelTellState(self.state)
        
        self.ctrl_dt = 1 / self.ctrl_freq
        self.firmware_dt = 1 / self.firmware_freq
        
        # Initialize visualization tools 
        self.first_motor_killed_print = True

        return init_obs, init_info

    def step(self, i):
        '''Step the firmware_wrapper class and its environment. 
        This function should be called once at the rate of ctrl_freq. Step processes and high level commands, 
        and runs the firmware loop and simulator according to the frequencies set. 

        Args: 
            i: the simulation control step index
        Todo:
            * Add support for state estimation 
        '''
        t = i / self.ctrl_freq

        self._process_command_queue(t)

        while self.tick / self.firmware_freq < t + self.ctrl_dt:
            # Step the environment and print all returned information.
            obs, reward, terminated, truncated, info = super().step(self.action)

            # Get state values from pybullet
            cur_pos=np.array([obs[0][0], obs[0][1], obs[0][2]]) # global coord, m
            cur_vel=np.array([obs[0][10], obs[0][11], obs[0][12]]) # global coord, m/s
            cur_rpy = np.array([obs[0][7], obs[0][8], obs[0][9]]) # body coord, rad 
            body_rot = R.from_euler('XYZ', cur_rpy).inv()

            if self.takeoff_sent:
                self.states += [[self.tick / self.firmware_freq, cur_pos[0], cur_pos[1], cur_pos[2]]]

            # Estimate rates 
            cur_rotation_rates = (cur_rpy - self.prev_rpy) / self.firmware_dt # body coord, rad/s
            self.prev_rpy = cur_rpy
            cur_acc = (cur_vel - self.prev_vel) / self.firmware_dt / 9.8 + np.array([0, 0, 1]) # global coord
            self.prev_vel = cur_vel
            
            # Update state 
            state_timestamp = int(self.tick / self.firmware_freq * 1e3)
            if self.STATE_DELAY:
                raise NotImplementedError("State delay is not yet implemented. Leave at 0.")
                self._update_state(state_timestamp, *self.state_history[0])
                self.state_history = self.state_history[1:] + [[cur_pos, cur_vel, cur_acc, cur_rpy * self.RAD_TO_DEG]]
            else:
                self._update_state(state_timestamp, cur_pos, cur_vel, cur_acc, cur_rpy * self.RAD_TO_DEG)#, quat=cur_quat)

            # Update sensor data 
            sensor_timestamp = int(self.tick / self.firmware_freq * 1e6)
            if self.SENSOR_DELAY:
                self._update_sensorData(sensor_timestamp, *self.sensor_history[0])
                self.sensor_history = self.sensor_history[1:] + [[body_rot.apply(cur_acc), cur_rotation_rates * self.RAD_TO_DEG]]
            else:
                self._update_sensorData(sensor_timestamp, body_rot.apply(cur_acc), cur_rotation_rates * self.RAD_TO_DEG)

            # Update setpoint 
            self._updateSetpoint(self.tick / self.firmware_freq) # setpoint looks right 

            # Step controller 
            self._step_controller()

            # Get action 
            new_action = self.PWM2RPM_SCALE * np.clip(np.array(self.pwms), self.MIN_PWM, self.MAX_PWM) + self.PWM2RPM_CONST

            if self.ACTION_DELAY:
                # Delays action commands to mimic real life hardware response delay 
                action = self.action_history[0]
                self.action_history = self.action_history[1:] + [new_action]
            else:
                action = new_action

            if self._error:
                action = np.zeros(4)
                if self.first_motor_killed_print:
                    print("Drone firmware error. Motors are killed.")
                    self.first_motor_killed_print = False

            self.action = action 

        return obs, reward, terminated, truncated, info
    
    def _update_initial_state(self, obs):
        self.prev_vel = np.array([obs[10], obs[11], obs[12]])
        self.prev_rpy = np.array([obs[7], obs[8], obs[9]])


    ##################################
    ########## Sensor Data ###########
    ##################################

    def _update_sensorData(self, timestamp, acc_vals, gyro_vals, baro_vals=[1013.25, 25]):
        '''
            Axis3f acc;               // Gs
            Axis3f gyro;              // deg/s
            Axis3f mag;               // gauss
            baro_t baro;              // C, Pa
            #ifdef LOG_SEC_IMU
                Axis3f accSec;            // Gs
                Axis3f gyroSec;           // deg/s
            #endif
            uint64_t interruptTimestamp;   // microseconds 
        '''
        ## Only gyro and acc are used in controller. Mag and baro used in state etimation (not yet supported)
        self._update_acc(*acc_vals)
        self._update_gyro(*gyro_vals)
        # self._update_gyro(self.sensorData.mag, *mag_vals)
        # self._update_baro(self.sensorData.baro, *baro_vals)

        self.sensorData.interruptTimestamp = timestamp
        self.sensorData_set = True
    
    def _update_gyro(self, x, y, z):
        self.sensorData.gyro.x = firm.lpf2pApply(self.gyrolpf[0], x)
        self.sensorData.gyro.y = firm.lpf2pApply(self.gyrolpf[1], y)
        self.sensorData.gyro.z = firm.lpf2pApply(self.gyrolpf[2], z)
        
    def _update_acc(self, x, y, z):
        self.sensorData.acc.x = firm.lpf2pApply(self.acclpf[0], x)
        self.sensorData.acc.y = firm.lpf2pApply(self.acclpf[1], y)
        self.sensorData.acc.z = firm.lpf2pApply(self.acclpf[2], z)

    def _update_baro(self, baro, pressure, temperature):
        '''
        pressure: hPa 
        temp: C
        asl = m 
        '''
        baro.pressure = pressure #* 0.01 Best guess is this is because the sensor encodes raw reading two decimal places and stores as int 
        baro.temperature = temperature
        baro.asl = (((1015.7 / baro.pressure)**0.1902630958 - 1) * (25 + 273.15)) / 0.0065
    

    ##################################
    ######### State Update ###########
    ##################################

    def _update_state(self, timestamp, pos, vel, acc, rpy, quat=None):
        '''
            attitude_t attitude;      // deg (legacy CF2 body coordinate system, where pitch is inverted)
            quaternion_t attitudeQuaternion;
            point_t position;         // m
            velocity_t velocity;      // m/s
            acc_t acc;                // Gs (but acc.z without considering gravity)
        '''
        self._update_attitude_t(self.state.attitude, timestamp, *rpy) # RPY required for PID and high level commander
        if self.CONTROLLER == 'mellinger':
            self._update_attitudeQuaternion(self.state.attitudeQuaternion, timestamp, *rpy) # Quat required for Mellinger 

        self._update_3D_vec(self.state.position, timestamp, *pos)
        self._update_3D_vec(self.state.velocity, timestamp, *vel)
        self._update_3D_vec(self.state.acc, timestamp, *acc)
        self.state_set = True

    def _update_3D_vec(self, point, timestamp, x, y, z):
        point.x = x
        point.y = y
        point.z = z
        point.timestamp = timestamp

    def _update_attitudeQuaternion(self, quaternion_t, timestamp, qx, qy, qz, qw=None):
        '''Updates attitude quaternion.

        Note:
            if qw is present, input is taken as a quat. Else, as roll, pitch, and yaw in deg
        '''
        quaternion_t.timestamp = timestamp

        if qw is None: # passed roll, pitch, yaw 
            qx, qy, qz, qw = _get_quaternion_from_euler(qx/self.RAD_TO_DEG, qy/self.RAD_TO_DEG, qz/self.RAD_TO_DEG) 

        quaternion_t.x = qx
        quaternion_t.y = qy
        quaternion_t.z = qz
        quaternion_t.w = qw

    def _update_attitude_t(self, attitude_t, timestamp, roll, pitch, yaw):
        attitude_t.timestamp = timestamp
        attitude_t.roll = roll
        attitude_t.pitch = -pitch # Legacy representation in CF firmware
        attitude_t.yaw = yaw


    ##################################
    ########### Controller ###########
    ################################## 

    def _step_controller(self):
        if not (self.sensorData_set):
            print("WARNING: sensorData has not been updated since last controller call.")
        if not (self.state_set):
            print("WARNING: state has not been updated since last controller call.")
        self.sensorData_set = False
        self.state_set = False

        # Check for tumbling crazyflie 
        if self.state.acc.z < -0.5: 
            self.tumble_counter += 1
        else:
            self.tumble_counter = 0
        if self.tumble_counter >= 30:
            print('WARNING: CrazyFlie is Tumbling. Killing motors to save propellers.')
            self.pwms = [0, 0, 0, 0]
            self.tick += 1
            self._error = True
            return 

        # Determine tick based on time passed, allowing us to run pid slower than the 1000Hz it was designed for
        cur_time = self.tick / self.firmware_freq
        if (cur_time - self.last_att_pid_call > 0.002) and (cur_time - self.last_pos_pid_call > 0.01):
            _tick = 0 # Runs position and attitude controller
            self.last_pos_pid_call = cur_time
            self.last_att_pid_call = cur_time
        elif (cur_time - self.last_att_pid_call > 0.002):
            self.last_att_pid_call = cur_time
            _tick = 2 # Runs attitude controller 
        else:
            _tick = 1 # Runs neither controller 

        # Step the chosen controller 
        if self.CONTROLLER == 'pid':
            firm.controllerPid(
                self.control,
                self.setpoint,
                self.sensorData,
                self.state,
                _tick
            )
        elif self.CONTROLLER == 'mellinger':
            firm.controllerMellinger(
                self.control,
                self.setpoint,
                self.sensorData,
                self.state,
                _tick
            )

        # Get pwm values from control object 
        self._powerDistribution(self.control)
        self.tick += 1

    def _updateSetpoint(self, timestep):
        if not self.full_state_cmd_override:
            firm.crtpCommanderHighLevelTellState(self.state)
            firm.crtpCommanderHighLevelUpdateTime(timestep) # Sets commander time variable --- this is time in s from start of flight 
            firm.crtpCommanderHighLevelGetSetpoint(self.setpoint, self.state)

    def _process_command_queue(self, sim_time):
        if len(self.command_queue) > 0:
            firm.crtpCommanderHighLevelStop() # Resets planner object        
            firm.crtpCommanderHighLevelUpdateTime(sim_time) # Sets commander time variable --- this is time in s from start of flight 
            command, args = self.command_queue.pop(0)
            getattr(self, command)(*args)

    def sendFullStateCmd(self, pos, vel, acc, yaw, rpy_rate, timestep):
        """Adds a sendfullstate command to command processing queue. 
        
        Notes:
            Overrides any high level commands being processed. 

        Args:
            pos (list): [x, y, z] position of the CF (m) 
            vel (list): [x, y, z] velocity of the CF (m/s)
            acc (list): [x, y, z] acceleration of the CF (m/s^2)
            yaw (float): yaw of the CF (rad)
            rpy_rate (list): roll, pitch, yaw rates (rad/s)
            timestep (float): simulation time when command is sent (s)
        """
        self.command_queue += [['_sendFullStateCmd', [pos, vel, acc, yaw, rpy_rate, timestep]]]

    def _sendFullStateCmd(self, pos, vel, acc, yaw, rpy_rate, timestep):
        # print(f"INFO_{self.tick}: Full state command sent.")
        self.setpoint.position.x = pos[0]
        self.setpoint.position.y = pos[1]
        self.setpoint.position.z = pos[2]
        self.setpoint.velocity.x = vel[0]
        self.setpoint.velocity.y = vel[1]
        self.setpoint.velocity.z = vel[2]
        self.setpoint.acceleration.x = acc[0]
        self.setpoint.acceleration.y = acc[1]
        self.setpoint.acceleration.z = acc[2]

        self.setpoint.attitudeRate.roll = rpy_rate[0] * self.RAD_TO_DEG
        self.setpoint.attitudeRate.pitch = rpy_rate[1] * self.RAD_TO_DEG
        self.setpoint.attitudeRate.yaw = rpy_rate[2] * self.RAD_TO_DEG

        quat = _get_quaternion_from_euler(0, 0, yaw)
        self.setpoint.attitudeQuaternion.x = quat[0]
        self.setpoint.attitudeQuaternion.y = quat[1]
        self.setpoint.attitudeQuaternion.z = quat[2]
        self.setpoint.attitudeQuaternion.w = quat[3]

        # self.setpoint.attitude.yaw = yaw * 180 / math.pi
        # self.setpoint.attitude.pitch = 0
        # self.setpoint.attitude.roll = 0

        # initilize setpoint modes to match cmdFullState 
        self.setpoint.mode.x = firm.modeAbs
        self.setpoint.mode.y = firm.modeAbs
        self.setpoint.mode.z = firm.modeAbs

        self.setpoint.mode.quat = firm.modeAbs
        self.setpoint.mode.roll = firm.modeDisable
        self.setpoint.mode.pitch = firm.modeDisable
        self.setpoint.mode.yaw = firm.modeDisable

        self.setpoint.timestamp = int(timestep*1000) # TODO: This may end up skipping control loops 
        self.full_state_cmd_override = True

    def sendTakeoffCmd(self, height, duration):
        """Adds a takeoff command to command processing queue. 

        Args:
            height (float): target takeoff height (m) 
            duration: (float): length of manuever
        """
        self.command_queue += [['_sendTakeoffCmd', [height, duration]]]
    def _sendTakeoffCmd(self, height, duration):
        print(f"INFO_{self.tick}: Takeoff command sent.")
        self.takeoff_sent = True
        firm.crtpCommanderHighLevelTakeoff(height, duration)
        self.full_state_cmd_override = False

    def sendTakeoffYawCmd(self, height, duration, yaw):
        """Adds a takeoffyaw command to command processing queue. 

        Args:
            height (float): target takeoff height (m) 
            duration: (float): length of manuever
            yaw (float): target yaw (rad)
        """
        self.command_queue += [['_sendTakeoffYawCmd', [height, duration, yaw]]]
    def _sendTakeoffYawCmd(self, height, duration, yaw):
        print(f"INFO_{self.tick}: Takeoff command sent.")
        firm.crtpCommanderHighLevelTakeoffYaw(height, duration, yaw)
        self.full_state_cmd_override = False

    def sendTakeoffVelCmd(self, height, vel, relative):
        """Adds a takeoffvel command to command processing queue. 

        Args:
            height (float): target takeoff height (m) 
            vel (float): target takeoff velocity (m/s)
            relative: (bool): whether takeoff height is relative to CF's current position
        """
        self.command_queue += [['_sendTakeoffVelCmd', [height, vel, relative]]]
    def _sendTakeoffVelCmd(self, height, vel, relative):
        print(f"INFO_{self.tick}: Takeoff command sent.")
        firm.crtpCommanderHighLevelTakeoffWithVelocity(height, vel, relative)
        self.full_state_cmd_override = False

    def sendLandCmd(self, height, duration):
        """Adds a land command to command processing queue. 

        Args:
            height (float): target landing height (m) 
            duration: (float): length of manuever
        """
        self.command_queue += [['_sendLandCmd', [height, duration]]]
    def _sendLandCmd(self, height, duration):
        print(f"INFO_{self.tick}: Land command sent.")
        firm.crtpCommanderHighLevelLand(height, duration)
        self.full_state_cmd_override = False

    def sendLandYawCmd(self, height, duration, yaw):
        """Adds a landyaw command to command processing queue. 

        Args:
            height (float): target landing height (m) 
            duration: (float): length of manuever
            yaw (float): target yaw (rad)
        """
        self.command_queue += [['_sendLandYawCmd', [height, duration, yaw]]]
    def _sendLandYawCmd(self, height, duration, yaw):
        print(f"INFO_{self.tick}: Land command sent.")
        firm.crtpCommanderHighLevelLandYaw(height, duration, yaw)
        self.full_state_cmd_override = False

    def sendLandVelCmd(self, height, vel, relative):
        """Adds a landvel command to command processing queue. 

        Args:
            height (float): target landing height (m) 
            vel (float): target landing velocity (m/s)
            relative: (bool): whether landing height is relative to CF's current position
        """
        self.command_queue += [['_sendLandVelCmd', [height, vel, relative]]]
    def _sendLandVelCmd(self, height, vel, relative):
        print(f"INFO_{self.tick}: Land command sent.")
        firm.crtpCommanderHighLevelLandWithVelocity(height, vel, relative)
        self.full_state_cmd_override = False

    def sendStopCmd(self):
        """Adds a stop command to command processing queue. 
        """
        self.command_queue += [['_sendStopCmd', []]]
    def _sendStopCmd(self):
        print(f"INFO_{self.tick}: Stop command sent.")
        firm.crtpCommanderHighLevelStop()
        self.full_state_cmd_override = False
        
    def sendGotoCmd(self, pos, yaw, duration_s, relative):
        """Adds a goto command to command processing queue. 

        Args:
            pos (list): [x, y, z] target position (m)
            yaw (float): target yaw (rad)
            duration_s (float): length of manuever
            relative (bool): whether setpoint is relative to CF's current position 
        """
        self.command_queue += [['_sendGotoCmd', [pos, yaw, duration_s, relative]]]
    def _sendGotoCmd(self, pos, yaw, duration_s, relative):
        print(f"INFO_{self.tick}: Go to command sent.")
        firm.crtpCommanderHighLevelGoTo(*pos, yaw, duration_s, relative)
        self.full_state_cmd_override = False

    def notifySetpointStop(self):
        """Adds a notifySetpointStop command to command processing queue. 
        """
        self.command_queue += [['_notifySetpointStop', []]]
    def _notifySetpointStop(self):
        """Adds a notifySetpointStop command to command processing queue. 
        """
        print(f"INFO_{self.tick}: Notify setpoint stop command sent.")
        firm.crtpCommanderHighLevelTellState(self.state)
        self.full_state_cmd_override = False


    ##################################
    ###### Hardware Functions ########
    ##################################

    BRUSHED = True
    SUPPLY_VOLTAGE = 3
    def _motorsGetPWM(self, thrust):
        if (self.BRUSHED):
            thrust = thrust / 65536 * 60
            volts = -0.0006239 * thrust**2 + 0.088 * thrust
            percentage = min(1, volts / self.SUPPLY_VOLTAGE)
            ratio = percentage * self.MAX_PWM

            return ratio
        else: 
            raise NotImplementedError("Emulator does not support the brushless motor configuration at this time.")

    def _limitThrust(self, val):
        if val > self.MAX_PWM:
            return self.MAX_PWM
        elif val < 0:
            return 0
        return val

    def _powerDistribution(self, control_t):
        motor_pwms = []
        if self.QUAD_FORMATION_X:
            r = control_t.roll / 2
            p = control_t.pitch / 2

            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust - r + p + control_t.yaw))]
            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust - r - p - control_t.yaw))]
            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust + r - p + control_t.yaw))]
            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust + r + p - control_t.yaw))]
        else:
            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust + control_t.pitch + control_t.yaw))]
            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust - control_t.roll - control_t.yaw))]
            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust - control_t.pitch + control_t.yaw))]
            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust + control_t.roll - control_t.yaw))]
        
        if self.MOTOR_SET_ENABLE:
            self.pwms = motor_pwms
        else:
            self.pwms = np.clip(motor_pwms, self.MIN_PWM).tolist()
    

    ##################################
    ##### Base Aviary Overrides ######
    ##################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            An ndarray of shape (NUM_DRONES, 4) for the commanded RPMs.

        """
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([[0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        spaces.Box
            The observation space, i.e., an ndarray of shape (NUM_DRONES, 20).

        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array([[-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of the state, see the implementation of `_getDroneStateVector()`.

        Returns
        -------
        ndarray
            An ndarray of shape (NUM_DRONES, 20) with the state of each drone.

        """
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Clips and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : ndarray
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """

        return action.reshape((1,4))

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        return -1

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False
    
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
    
def _get_quaternion_from_euler(roll, pitch, yaw):
    """Convert an Euler angle to a quaternion.
    
    Args:
        roll (float): The roll (rotation around x-axis) angle in radians.
        pitch (float): The pitch (rotation around y-axis) angle in radians.
        yaw (float): The yaw (rotation around z-axis) angle in radians.
    
    Returns:
        list: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    
    return [qx, qy, qz, qw]


================================================
FILE: gym_pybullet_drones/envs/CtrlAviary.py
================================================
import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class CtrlAviary(BaseAviary):
    """Multi-drone environment class for control applications."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results'
                 ):
        """Initialization of an aviary environment for control applications.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.

        """
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder
                         )

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            An ndarray of shape (NUM_DRONES, 4) for the commanded RPMs.

        """
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([[0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        spaces.Box
            The observation space, i.e., an ndarray of shape (NUM_DRONES, 20).

        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array([[-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of the state, see the implementation of `_getDroneStateVector()`.

        Returns
        -------
        ndarray
            An ndarray of shape (NUM_DRONES, 20) with the state of each drone.

        """
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Clips and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : ndarray
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        return np.array([np.clip(action[i, :], 0, self.MAX_RPM) for i in range(self.NUM_DRONES)])

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        return -1

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False
    
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years



================================================
FILE: gym_pybullet_drones/envs/HoverAviary.py
================================================
import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class HoverAviary(BaseRLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.TARGET_POS = np.array([0,0,1])
        self.EPISODE_LEN_SEC = 8
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        ret = max(0, 2 - np.linalg.norm(self.TARGET_POS-state[0:3])**4)
        return ret

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .0001:
            return True
        else:
            return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0 # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years



================================================
FILE: gym_pybullet_drones/envs/MultiHoverAviary.py
================================================
import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class MultiHoverAviary(BaseRLAviary):
    """Multi-agent RL problem: leader-follower."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.EPISODE_LEN_SEC = 8
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obs=obs,
                         act=act
                         )
        self.TARGET_POS = self.INIT_XYZS + np.array([[0,0,1/(i+1)] for i in range(num_drones)])

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        ret = 0
        for i in range(self.NUM_DRONES):
            ret += max(0, 2 - np.linalg.norm(self.TARGET_POS[i,:]-states[i][0:3])**4)
        return ret

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        dist = 0
        for i in range(self.NUM_DRONES):
            dist += np.linalg.norm(self.TARGET_POS[i,:]-states[i][0:3])
        if dist < .0001:
            return True
        else:
            return False

    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            if (abs(states[i][0]) > 2.0 or abs(states[i][1]) > 2.0 or states[i][2] > 2.0 # Truncate when a drones is too far away
             or abs(states[i][7]) > .4 or abs(states[i][8]) > .4 # Truncate when a drone is too tilted
            ):
                return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years



================================================
FILE: gym_pybullet_drones/envs/VelocityAviary.py
================================================
import os
import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class VelocityAviary(BaseAviary):
    """Multi-drone environment class for high-level planning."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results'
                 ):
        """Initialization of an aviary environment for or high-level planning.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.

        """
        #### Create integrated controllers #########################
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
            self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder
                         )
        #### Set a limit on the maximum target speed ###############
        self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            An ndarray of shape (NUM_DRONES, 4) for the commanded velocity vectors.

        """
        #### Action vector ######### X       Y       Z   fract. of MAX_SPEED_KMH
        act_lower_bound = np.array([[-1,     -1,     -1,                        0] for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([[ 1,      1,      1,                        1] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        spaces.Box
            The observation space, i.e., and ndarray of shape (NUM_DRONES, 20).

        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array([[-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of the state, see the implementation of `_getDroneStateVector()`.

        Returns
        -------
        ndarray
            An ndarray of shape (NUM_DRONES, 20) with the state of each drone.

        """
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Uses PID control to target a desired velocity vector.

        Parameters
        ----------
        action : ndarray
            The desired velocity input for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        rpm = np.zeros((self.NUM_DRONES, 4))
        for k in range(action.shape[0]):
            #### Get the current state of the drone  ###################
            state = self._getDroneStateVector(k)
            target_v = action[k, :]
            #### Normalize the first 3 components of the target velocity
            if np.linalg.norm(target_v[0:3]) != 0:
                v_unit_vector = target_v[0:3] / np.linalg.norm(target_v[0:3])
            else:
                v_unit_vector = np.zeros(3)
            temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                    cur_pos=state[0:3],
                                                    cur_quat=state[3:7],
                                                    cur_vel=state[10:13],
                                                    cur_ang_vel=state[13:16],
                                                    target_pos=state[0:3], # same as the current position
                                                    target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                    target_vel=self.SPEED_LIMIT * np.abs(target_v[3]) * v_unit_vector # target the desired velocity vector
                                                    )
            rpm[k,:] = temp
        return rpm

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        return -1

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False
    
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years



================================================
FILE: gym_pybullet_drones/examples/__init__.py
================================================
[Empty file]


================================================
FILE: gym_pybullet_drones/examples/beta.py
================================================
"""Control + Betaflight. 

Setup
-----
Use script `gym_pybullet_drones/assets/clone_bfs.sh` to create
executables for as many drones as needed (e.g. 2):
    $ ./gym_pybullet_drones/assets/clone_bfs.sh 2

Note
-------
This example will automatically start as many SITL Betaflight as drones
in the simulation in separate terminals:

    $ cd betaflights/bf0/
    $ ./obj/main/betaflight_SITL.elf

    $ cd betaflights/bf1/
    $ ./obj/main/betaflight_SITL.elf

    $ ..

Run as:

    $ cd gym-pybullet-drones/gym_pybullet_drones/examples/
    $ python beta.py

"""
import time
import argparse
import numpy as np
import csv

from transforms3d.quaternions import rotate_vector, qconjugate, mat2quat, qmult
from transforms3d.utils import normalized_vector

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.BetaAviary import BetaAviary
from gym_pybullet_drones.control.CTBRControl import CTBRControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("racer")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_SIMULATION_FREQ_HZ = 500
DEFAULT_CONTROL_FREQ_HZ = 500
DEFAULT_DURATION_SEC = 20
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_NUM_DRONES = 2

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        ):
    #### Create the environment with or without video capture ##
    INIT_XYZ = np.array([[.3*i, .3*i, .1] for i in range(1,num_drones+1)])
    INIT_RPY = np.array([[.0, .0, .0] for _ in range(num_drones)])
    env = BetaAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZ,
                        initial_rpys=INIT_RPY,
                        physics=physics,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        user_debug_gui=user_debug_gui
                        )

    ctrl = CTBRControl(drone_model=drone)

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    )

    #### Run the simulation ####################################
    with open("../assets/beta-traj.csv", mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        trajectory1 = iter([{
            "pos": np.array([
                float(row["p_x"]),
                float(row["p_y"]),
                float(row["p_z"]),
            ]),
            "vel": np.array([
                float(row["v_x"]),
                float(row["v_y"]),
                float(row["v_z"]),
            ]),
        } for row in csv_reader])
    with open("../assets/beta-traj.csv", mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        trajectory2 = iter(reversed([{
            "pos": np.array([
                float(row["p_x"]),
                float(row["p_y"]),
                float(row["p_z"]),
            ]),
            "vel": np.array([
                float(row["v_x"]),
                float(row["v_y"]),
                float(row["v_z"]),
            ]),
        } for row in csv_reader]))
    action = np.zeros((num_drones,4))
    ARM_TIME = 1.
    TRAJ_TIME = 1.5
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        t = i/env.CTRL_FREQ
        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action, i)
        
        if t > env.TRAJ_TIME:
            for j in range(num_drones):
                try:
                    target = next(trajectory1) if j%2 == 0 else next(trajectory2)
                    action[j,:] = ctrl.computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                    state=obs[j],
                                                    target_pos=target["pos"]+[INIT_XYZ[j][0], INIT_XYZ[j][1], 0],
                                                    target_vel=target["vel"]
                                                    )
                except:
                    break
        

        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(drone=j,
                        timestamp=i/env.CTRL_FREQ,
                        state=obs[j]
                        )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            pass
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("beta") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Test flight script using SITL Betaflight')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: BETA)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))



================================================
FILE: gym_pybullet_drones/examples/cf.py
================================================
"""CrazyFlie software-in-the-loop control example. 

Setup
-----
Step 1: Clone pycffirmware from https://github.com/utiasDSL/pycffirmware
Step 2: Follow the install instructions for pycffirmware in its README 

Example
-------
In terminal, run: 
python gym_pybullet_drones/examples/cf.py

"""
import time
import argparse
import numpy as np
import csv

from transforms3d.quaternions import rotate_vector, qconjugate, mat2quat, qmult
from transforms3d.utils import normalized_vector

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CFAviary import CFAviary
from gym_pybullet_drones.control.CTBRControl import CTBRControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_SIMULATION_FREQ_HZ = 500
DEFAULT_CONTROL_FREQ_HZ = 25
DEFAULT_OUTPUT_FOLDER = 'results'
NUM_DRONES = 1
INIT_XYZ = np.array([[.5*i, .5*i, .1] for i in range(NUM_DRONES)])
INIT_RPY = np.array([[.0, .0, .0] for _ in range(NUM_DRONES)])

def run(
        drone=DEFAULT_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        ):
    #### Create the environment with or without video capture ##
    env = CFAviary(drone_model=drone,
                        num_drones=NUM_DRONES,
                        initial_xyzs=INIT_XYZ,
                        initial_rpys=INIT_RPY,
                        physics=physics,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        user_debug_gui=user_debug_gui
                        )

    # ctrl = CTBRControl(drone_model=drone)

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=NUM_DRONES,
                    output_folder=output_folder,
                    )

    #### Run the simulation ####################################
    delta = 75 # 3s @ 25hz control loop 
    trajectory = [[0, 0, 0] for i in range(delta)] + \
        [[0, 0, i/delta] for i in range(delta)] + \
        [[i/delta, 0, 1] for i in range(delta)] + \
        [[1, i/delta, 1] for i in range(delta)] + \
        [[1-i/delta, 1, 1] for i in range(delta)] + \
        [[0, 1-i/delta, 1] for i in range(delta)] + \
        [[0, 0, 1-i/delta] for i in range(delta)]

    START = time.time()
    for i in range(0, len(trajectory)):
        t = i/env.ctrl_freq
        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(i)
        
        for j in range(NUM_DRONES):
            try:
                target = trajectory[i]
                pos = target+[INIT_XYZ[j][0], INIT_XYZ[j][1], 0]
                vel = np.zeros(3)
                acc = np.zeros(3)
                yaw = i*np.pi/delta/2
                rpy_rate = np.zeros(3)
                env.sendFullStateCmd(pos, vel, acc, yaw, rpy_rate, t)
            except:
                break

        #### Log the simulation ####################################
        for j in range(NUM_DRONES):
            logger.log(drone=j,
                        timestamp=i/env.CTRL_FREQ,
                        state=obs[j]
                        )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            pass
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("beta") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Test flight script using SITL Betaflight')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: BETA)', metavar='', choices=DroneModel)
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 500)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 25)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))



================================================
FILE: gym_pybullet_drones/examples/debug.py
================================================
import time
import numpy as np
import pybullet as p
import pybullet_data

DURATION_SEC = 5

X_AX_1 = -1; Y_AX_1 = -1; Z_AX_1 = -1;
X_AX_2 = -1; Y_AX_2 = -1; Z_AX_2 = -1; 
TEXT = -1


if __name__ == "__main__":

    PYB_CLIENT = p.connect(p.GUI, key=0); p.setRealTimeSimulation(0, physicsClientId=PYB_CLIENT); p.setTimeStep(1/240, physicsClientId=PYB_CLIENT)
    p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=-30, cameraPitch=-30, cameraTargetPosition=[0.0,0.0,0.0], physicsClientId=PYB_CLIENT)
    p.setGravity(0, 0, 0, physicsClientId=PYB_CLIENT) # Make objects weightless

    ROBOT_1 = p.loadURDF("../assets/box.urdf",[0,0,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=PYB_CLIENT)
    ROBOT_2 = p.loadURDF("../assets/box.urdf",[0,0,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=PYB_CLIENT)
    
    p.resetBasePositionAndOrientation(ROBOT_1,
                                      posObj=[-.2,-.2,.5], 
                                      ornObj=p.getQuaternionFromEuler(
                                                [45*(np.pi/180),0*(np.pi/180),0*(np.pi/180)],
                                                physicsClientId=PYB_CLIENT), physicsClientId=PYB_CLIENT)
    p.resetBasePositionAndOrientation(ROBOT_2,
                                      posObj=[1.,.0,.0],
                                      ornObj=p.getQuaternionFromEuler(
                                                [0*(np.pi/180),0*(np.pi/180),0*(np.pi/180)],
                                                physicsClientId=PYB_CLIENT), physicsClientId=PYB_CLIENT)

    # VIDEO = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4, fileName="~/Desktop/video.mp4", physicsClientId=PYB_CLIENT)
    START = time.time()
    for i in range(DURATION_SEC*240):

        X_AX_1 = p.addUserDebugLine(lineFromXYZ=[0,0,0],lineToXYZ=[0.1,0,0],lineColorRGB=[1,0,0],parentObjectUniqueId=ROBOT_1, parentLinkIndex=-1, replaceItemUniqueId=X_AX_1, physicsClientId=PYB_CLIENT)
        Y_AX_1 = p.addUserDebugLine(lineFromXYZ=[0,0,0],lineToXYZ=[0,0.1,0],lineColorRGB=[0,1,0],parentObjectUniqueId=ROBOT_1, parentLinkIndex=-1, replaceItemUniqueId=Y_AX_1, physicsClientId=PYB_CLIENT)
        Z_AX_1 = p.addUserDebugLine(lineFromXYZ=[0,0,0],lineToXYZ=[0,0,0.1],lineColorRGB=[0,0,1],parentObjectUniqueId=ROBOT_1, parentLinkIndex=-1, replaceItemUniqueId=Z_AX_1, physicsClientId=PYB_CLIENT)
        
        X_AX_2 = p.addUserDebugLine(lineFromXYZ=[0,0,0],lineToXYZ=[0.1,0,0],lineColorRGB=[1,0,0],parentObjectUniqueId=ROBOT_2, parentLinkIndex=-1, replaceItemUniqueId=X_AX_2, physicsClientId=PYB_CLIENT)
        Y_AX_2 = p.addUserDebugLine(lineFromXYZ=[0,0,0],lineToXYZ=[0,0.1,0],lineColorRGB=[0,1,0],parentObjectUniqueId=ROBOT_2, parentLinkIndex=-1, replaceItemUniqueId=Y_AX_2, physicsClientId=PYB_CLIENT)
        Z_AX_2 = p.addUserDebugLine(lineFromXYZ=[0,0,0],lineToXYZ=[0,0,0.1],lineColorRGB=[0,0,1],parentObjectUniqueId=ROBOT_2, parentLinkIndex=-1, replaceItemUniqueId=Z_AX_2, physicsClientId=PYB_CLIENT)

        # p.applyExternalTorque(ROBOT_2, -1,
        #                       torqueObj=[
        #                                     0., # 5e-5
        #                                     0., # 5e-5
        #                                     0.  # 5e-5
        #                                 ],
        #                       flags=p.LINK_FRAME, # p.WORLD_FRAME
        #                       physicsClientId=PYB_CLIENT)
        # p.applyExternalForce(ROBOT_2, -1,
        #                      forceObj=[
        #                                 0., # 5e-2,
        #                                 0., # 5e-2,
        #                                 0.  # 5e-2
        #                             ],
        #                      posObj=[0.,0.,0.],
        #                      flags=p.LINK_FRAME, # p.WORLD_FRAME
        #                      physicsClientId=PYB_CLIENT) 
        
        p.applyExternalTorque(ROBOT_1, -1,
                              torqueObj=[
                                            0., # 5e-5
                                            0., # 5e-5
                                            5e-5  # 5e-5
                                        ],
                              flags=p.LINK_FRAME, # p.WORLD_FRAME
                              physicsClientId=PYB_CLIENT)
        p.applyExternalForce(ROBOT_1, -1,
                             forceObj=[
                                        0., # 5e-2,
                                        0., # 5e-2,
                                        0.  # 5e-2
                                    ],
                             posObj=[0.,0.,0.],
                             flags=p.LINK_FRAME, # p.WORLD_FRAME
                             physicsClientId=PYB_CLIENT)

        pos, quat = p.getBasePositionAndOrientation(ROBOT_1, physicsClientId=PYB_CLIENT)
        rpj = p.getEulerFromQuaternion(quat, physicsClientId=PYB_CLIENT)
        vel, ang_vel = p.getBaseVelocity(ROBOT_1, physicsClientId=PYB_CLIENT)
    
        # invPos, invOrn = p.invertTransform(pos, quat)
        # pos2, quat2 = p.getBasePositionAndOrientation(ROBOT_2, physicsClientId=PYB_CLIENT)
        # rel_pos, rel_or = p.multiplyTransforms(invPos, invOrn, pos2, quat2)

        TEXT = p.addUserDebugText(f'{vel}\n {ang_vel}', 
                                    textPosition=[-1.25,0,1], 
                                    textColorRGB=[0,0,1], 
                                    lifeTime=1, 
                                    textSize=1, 
                                    replaceItemUniqueId=TEXT, physicsClientId=PYB_CLIENT)

        ########################################################################################################################

        p.stepSimulation(physicsClientId=PYB_CLIENT)
        elapsed = time.time()-START
        if elapsed < i/240: time.sleep(i/240-elapsed)    

    # p.stopStateLogging(VIDEO, physicsClientId=PYB_CLIENT)






================================================
FILE: gym_pybullet_drones/examples/downwash.py
================================================
"""Script demonstrating the implementation of the downwash effect model.

Example
-------
In a terminal, run as:

    $ python downwash.py

Notes
-----
The drones move along 2D trajectories in the X-Z plane, between x == +.5 and -.5.

"""
import time
import argparse
import numpy as np

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger

DEFAULT_DRONE = DroneModel('cf2x')
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run(
        drone=DEFAULT_DRONE, 
        gui=DEFAULT_GUI, 
        record_video=DEFAULT_RECORD_VIDEO, 
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ, 
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ, 
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        plot=True,
        colab=DEFAULT_COLAB
    ):
    #### Initialize the simulation #############################
    INIT_XYZS = np.array([[.5, 0, 1],[-.5, 0, .5]])
    env = CtrlAviary(drone_model=drone,
                     num_drones=2,
                     initial_xyzs=INIT_XYZS,
                     physics=Physics.PYB_DW,
                     neighbourhood_radius=10,
                     pyb_freq=simulation_freq_hz,
                     ctrl_freq=control_freq_hz,
                     gui=gui,
                     record=record_video,
                     obstacles=True
                     )

    #### Initialize the trajectories ###########################
    PERIOD = 5
    NUM_WP = control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP, 2))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = [0.5*np.cos(2*np.pi*(i/NUM_WP)), 0]
    wp_counters = np.array([0, int(NUM_WP/2)])

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=2,
                    duration_sec=duration_sec,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    ctrl = [DSLPIDControl(drone_model=drone) for i in range(2)]

    #### Run the simulation ####################################
    action = np.zeros((2,4))
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control for the current way point #############
        for j in range(2):
            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                    state=obs[j],
                                                                    target_pos=np.hstack([TARGET_POS[wp_counters[j], :], INIT_XYZS[j, 2]]),
                                                                    )

        #### Go to the next way point and loop #####################
        for j in range(2):
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        #### Log the simulation ####################################
        for j in range(2):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack([TARGET_POS[wp_counters[j], :], INIT_XYZS[j ,2], np.zeros(9)])
                       )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("dw") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Downwash example script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONE,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 10)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))


================================================
FILE: gym_pybullet_drones/examples/learn.py
================================================
"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('one_d_rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 2
DEFAULT_MA = False

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):

    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    if not multiagent:
        train_env = make_vec_env(HoverAviary,
                                 env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=1,
                                 seed=0
                                 )
        eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        train_env = make_vec_env(MultiHoverAviary,
                                 env_kwargs=dict(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=1,
                                 seed=0
                                 )
        eval_env = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Train the model #######################################
    model = PPO('MlpPolicy',
                train_env,
                # tensorboard_log=filename+'/tb/',
                verbose=1)

    #### Target cumulative rewards (problem-dependent) ##########
    if DEFAULT_ACT == ActionType.ONE_D_RPM:
        target_reward = 474. if not multiagent else 949.5
    else:
        target_reward = 467. if not multiagent else 920.
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    model.learn(total_timesteps=int(1e7) if local else int(1e2), # shorter training in GitHub Actions pytest
                callback=eval_callback,
                log_interval=100)

    #### Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        timesteps = data['timesteps']
        results = data['results'][:, 0] 
        print("Data from evaluations.npz")
        for j in range(timesteps.shape[0]):
            print(f"{timesteps[j]},{results[j]}")
        if local:
            plt.plot(timesteps, results, marker='o', linestyle='-', markersize=4)
            plt.xlabel('Training Steps')
            plt.ylabel('Episode Reward')
            plt.grid(True, alpha=0.6)
            plt.show()

    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################

    # if os.path.isfile(filename+'/final_model.zip'):
    #     path = filename+'/final_model.zip'
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)

    #### Show (and record a video of) the model's performance ##
    if not multiagent:
        test_env = HoverAviary(gui=gui,
                               obs=DEFAULT_OBS,
                               act=DEFAULT_ACT,
                               record=record_video)
        test_env_nogui = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        test_env = MultiHoverAviary(gui=gui,
                                        num_drones=DEFAULT_AGENTS,
                                        obs=DEFAULT_OBS,
                                        act=DEFAULT_ACT,
                                        record=record_video)
        test_env_nogui = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=DEFAULT_AGENTS if multiagent else 1,
                output_folder=output_folder,
                colab=colab
                )

    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                logger.log(drone=0,
                    timestamp=i/test_env.CTRL_FREQ,
                    state=np.hstack([obs2[0:3],
                                        np.zeros(4),
                                        obs2[3:15],
                                        act2
                                        ]),
                    control=np.zeros(12)
                    )
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(drone=d,
                        timestamp=i/test_env.CTRL_FREQ,
                        state=np.hstack([obs2[d][0:3],
                                            np.zeros(4),
                                            obs2[d][3:15],
                                            act2[d]
                                            ]),
                        control=np.zeros(12)
                        )
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))



================================================
FILE: gym_pybullet_drones/examples/mrac.py
================================================
import time

import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import pybullet_data

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.MRAC import MRAC
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 120
DEFAULT_DURATION_SEC = 15
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):

    INIT_XYZS = np.array([[0, 0, 0]])
    INIT_RPYS = np.array([[0, 0, 0]])

    TARGET_POS = np.array([[0, 0, 1]])
    TARGET_RPY = np.array([[0, 0, 0]])

    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )
    
    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )
    
    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P, DroneModel.RACE]:
        ctrl = [MRAC(drone_model=drone) for i in range(num_drones)]

    
    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        obs, reward, terminated, truncated, info = env.step(action)

        for j in range(num_drones):
            action[j,:], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                state=obs[j],
                                                                target_pos=TARGET_POS[j, :],
                                                                target_rpy=TARGET_RPY[j, :]
                                                                )
            
        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(drone=j,
                    timestamp=i/env.CTRL_FREQ,
                    state=obs[j],
                    control=np.hstack([TARGET_POS[j, :], TARGET_RPY[j, :], np.zeros(6)])
                    )
            
        env.render()

        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    env.close()

    logger.save()
    logger.save_as_csv("mrac")

    if plot:
        logger.plot()


if __name__ == "__main__":
    run()



================================================
FILE: gym_pybullet_drones/examples/pid.py
================================================
"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 3
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .3
    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])

    #### Initialize a circular trajectory ######################
    PERIOD = 10
    NUM_WP = control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP,3))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], 0
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])

    #### Debug trajectory ######################################
    #### Uncomment alt. target_pos in .computeControlFromState()
    # INIT_XYZS = np.array([[.3 * i, 0, .1] for i in range(num_drones)])
    # INIT_RPYS = np.array([[0, 0,  i * (np.pi/3)/num_drones] for i in range(num_drones)])
    # NUM_WP = control_freq_hz*15
    # TARGET_POS = np.zeros((NUM_WP,3))
    # for i in range(NUM_WP):
    #     if i < NUM_WP/6:
    #         TARGET_POS[i, :] = (i*6)/NUM_WP, 0, 0.5*(i*6)/NUM_WP
    #     elif i < 2 * NUM_WP/6:
    #         TARGET_POS[i, :] = 1 - ((i-NUM_WP/6)*6)/NUM_WP, 0, 0.5 - 0.5*((i-NUM_WP/6)*6)/NUM_WP
    #     elif i < 3 * NUM_WP/6:
    #         TARGET_POS[i, :] = 0, ((i-2*NUM_WP/6)*6)/NUM_WP, 0.5*((i-2*NUM_WP/6)*6)/NUM_WP
    #     elif i < 4 * NUM_WP/6:
    #         TARGET_POS[i, :] = 0, 1 - ((i-3*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-3*NUM_WP/6)*6)/NUM_WP
    #     elif i < 5 * NUM_WP/6:
    #         TARGET_POS[i, :] = ((i-4*NUM_WP/6)*6)/NUM_WP, ((i-4*NUM_WP/6)*6)/NUM_WP, 0.5*((i-4*NUM_WP/6)*6)/NUM_WP
    #     elif i < 6 * NUM_WP/6:
    #         TARGET_POS[i, :] = 1 - ((i-5*NUM_WP/6)*6)/NUM_WP, 1 - ((i-5*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-5*NUM_WP/6)*6)/NUM_WP
    # wp_counters = np.array([0 for i in range(num_drones)])

    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### Make it rain rubber ducks #############################
        # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control for the current way point #############
        for j in range(num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                    state=obs[j],
                                                                    target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]]),
                                                                    # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                                                    target_rpy=INIT_RPYS[j, :]
                                                                    )

        #### Go to the next way point and loop #####################
        for j in range(num_drones):
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                       # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                       )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))



================================================
FILE: gym_pybullet_drones/examples/pid_velocity.py
================================================
"""Script demonstrating the joint use of velocity input.

The simulation is run by a `VelocityAviary` environment.

Example
-------
In a terminal, run as:

    $ python pid_velocity.py

Notes
-----
The drones use interal PID control to track a target velocity.

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary

DEFAULT_DRONE = DroneModel("cf2x")
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 5
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run(
        drone=DEFAULT_DRONE,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VIDEO,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
        #### Initialize the simulation #############################
    INIT_XYZS = np.array([
                          [ 0, 0, .1],
                          [.3, 0, .1],
                          [.6, 0, .1],
                          [0.9, 0, .1]
                          ])
    INIT_RPYS = np.array([
                          [0, 0, 0],
                          [0, 0, np.pi/3],
                          [0, 0, np.pi/4],
                          [0, 0, np.pi/2]
                          ])
    PHY = Physics.PYB

    #### Create the environment ################################
    env = VelocityAviary(drone_model=drone,
                         num_drones=4,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=Physics.PYB,
                         neighbourhood_radius=10,
                         pyb_freq=simulation_freq_hz,
                         ctrl_freq=control_freq_hz,
                         gui=gui,
                         record=record_video,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui
                         )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()
    DRONE_IDS = env.getDroneIds()

    #### Compute number of control steps in the simlation ######
    PERIOD = duration_sec
    NUM_WP = control_freq_hz*PERIOD
    wp_counters = np.array([0 for i in range(4)])

    #### Initialize the velocity target ########################
    TARGET_VEL = np.zeros((4,NUM_WP,4))
    for i in range(NUM_WP):
        TARGET_VEL[0, i, :] = [-0.5, 1, 0, 0.99] if i < (NUM_WP/8) else [0.5, -1, 0, 0.99]
        TARGET_VEL[1, i, :] = [0, 1, 0, 0.99] if i < (NUM_WP/8+NUM_WP/6) else [0, -1, 0, 0.99]
        TARGET_VEL[2, i, :] = [0.2, 1, 0.2, 0.99] if i < (NUM_WP/8+2*NUM_WP/6) else [-0.2, -1, -0.2, 0.99]
        TARGET_VEL[3, i, :] = [0, 1, 0.5, 0.99] if i < (NUM_WP/8+3*NUM_WP/6) else [0, -1, -0.5, 0.99]

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=4,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Run the simulation ####################################
    action = np.zeros((4,4))
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        ############################################################
        # for j in range(3): env._showDroneLocalAxes(j)

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control for the current way point #############
        for j in range(4):
            action[j, :] = TARGET_VEL[j, wp_counters[j], :] 

        #### Go to the next way point and loop #####################
        for j in range(4):
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        #### Log the simulation ####################################
        for j in range(4):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state= obs[j],
                       control=np.hstack([TARGET_VEL[j, wp_counters[j], 0:3], np.zeros(9)])
                       )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Plot the simulation results ###########################
    logger.save_as_csv("vel") # Optional CSV save
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Velocity control example using VelocityAviary')
    parser.add_argument('--drone',              default=DEFAULT_DRONE,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,      type=str2bool,      help='Whether to add obstacles to the environment (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))




================================================
FILE: gym_pybullet_drones/examples/play.py
================================================
import os
import time
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger

DEFAULT_MODEL_PATH = "results/best_model.zip"
DEFAULT_GUI = True
DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('one_d_rpm')
DEFAULT_AGENTS = 2
DEFAULT_MA = False

def play(model_path=DEFAULT_MODEL_PATH, multiagent=DEFAULT_MA, gui=DEFAULT_GUI):
    #### Load saved model ####
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        return

    model = PPO.load(model_path)
    print(f"[INFO] Loaded model from {model_path}")

    #### Create test environment ####
    if not multiagent:
        env = HoverAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        env = MultiHoverAviary(gui=gui, num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    logger = Logger(logging_freq_hz=int(env.CTRL_FREQ),
                    num_drones=DEFAULT_AGENTS if multiagent else 1,
                    output_folder="logs_playback/",
                    colab=False)

    #### Run the simulation ####
    obs, _ = env.reset(seed=42, options={})
    start = time.time()

    for i in range((env.EPISODE_LEN_SEC+2)*env.CTRL_FREQ):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        obs2 = obs.squeeze()
        act2 = action.squeeze()

        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                logger.log(drone=0,
                    timestamp=i/env.CTRL_FREQ,
                    state=np.hstack([obs2[0:3],
                                     np.zeros(4),
                                     obs2[3:15],
                                     act2]),
                    control=np.zeros(12))
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(drone=d,
                        timestamp=i/env.CTRL_FREQ,
                        state=np.hstack([obs2[d][0:3],
                                         np.zeros(4),
                                         obs2[d][3:15],
                                         act2[d]]),
                        control=np.zeros(12))

        env.render()
        sync(i, start, env.CTRL_TIMESTEP)
        if terminated:
            break

    env.close()
    logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained PPO policy in PyBullet drones environment.")
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to saved policy zip file')
    parser.add_argument('--multiagent', type=bool, default=DEFAULT_MA, help='Whether to use MultiHoverAviary')
    parser.add_argument('--gui', type=bool, default=DEFAULT_GUI, help='Enable GUI rendering')
    args = parser.parse_args()

    play(**vars(args))



================================================
FILE: gym_pybullet_drones/utils/__init__.py
================================================
[Empty file]


================================================
FILE: gym_pybullet_drones/utils/enums.py
================================================
from enum import Enum

class DroneModel(Enum):
    """Drone models enumeration class."""

    CF2X = "cf2x"   # Bitcraze Craziflie 2.0 in the X configuration
    CF2P = "cf2p"   # Bitcraze Craziflie 2.0 in the + configuration
    RACE = "racer"  # Racer drone in the X configuration


################################################################################

class Physics(Enum):
    """Physics implementations enumeration class."""

    PYB = "pyb"                         # Base PyBullet physics update
    DYN = "dyn"                         # Explicit dynamics model
    PYB_GND = "pyb_gnd"                 # PyBullet physics update with ground effect
    PYB_DRAG = "pyb_drag"               # PyBullet physics update with drag
    PYB_DW = "pyb_dw"                   # PyBullet physics update with downwash
    PYB_GND_DRAG_DW = "pyb_gnd_drag_dw" # PyBullet physics update with ground effect, drag, and downwash

################################################################################

class ImageType(Enum):
    """Camera capture image type enumeration class."""

    RGB = 0     # Red, green, blue (and alpha)
    DEP = 1     # Depth
    SEG = 2     # Segmentation by object id
    BW = 3      # Black and white

################################################################################

class ActionType(Enum):
    """Action type enumeration class."""
    RPM = "rpm"                 # RPMS
    PID = "pid"                 # PID control
    VEL = "vel"                 # Velocity input (using PID control)
    ONE_D_RPM = "one_d_rpm"     # 1D (identical input to all motors) with RPMs
    ONE_D_PID = "one_d_pid"     # 1D (identical input to all motors) with PID control

################################################################################

class ObservationType(Enum):
    """Observation type enumeration class."""
    KIN = "kin"     # Kinematic information (pose, linear and angular velocities)
    RGB = "rgb"     # RGB camera capture in each drone's POV



================================================
FILE: gym_pybullet_drones/utils/Logger.py
================================================
import os
from datetime import datetime
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Logger(object):
    """A class for logging and visualization.

    Stores, saves to file, and plots the kinematic information and RPMs
    of a simulation with one or more drones.

    """

    ################################################################################

    def __init__(self,
                 logging_freq_hz: int,
                 output_folder: str="results",
                 num_drones: int=1,
                 duration_sec: int=0,
                 colab: bool=False,
                 ):
        """Logger class __init__ method.

        Note: the order in which information is stored by Logger.log() is not the same
        as the one in, e.g., the obs["id"]["state"], check the implementation below.

        Parameters
        ----------
        logging_freq_hz : int
            Logging frequency in Hz.
        num_drones : int, optional
            Number of drones.
        duration_sec : int, optional
            Used to preallocate the log arrays (improves performance).

        """
        self.COLAB = colab
        self.OUTPUT_FOLDER = output_folder
        if not os.path.exists(self.OUTPUT_FOLDER):
            os.mkdir(self.OUTPUT_FOLDER)
        self.LOGGING_FREQ_HZ = logging_freq_hz
        self.NUM_DRONES = num_drones
        self.PREALLOCATED_ARRAYS = False if duration_sec == 0 else True
        self.counters = np.zeros(num_drones)
        self.timestamps = np.zeros((num_drones, duration_sec*self.LOGGING_FREQ_HZ))
        #### Note: this is the suggest information to log ##############################
        self.states = np.zeros((num_drones, 16, duration_sec*self.LOGGING_FREQ_HZ)) #### 16 states: pos_x,
                                                                                                  # pos_y,
                                                                                                  # pos_z,
                                                                                                  # vel_x,
                                                                                                  # vel_y,
                                                                                                  # vel_z,
                                                                                                  # roll,
                                                                                                  # pitch,
                                                                                                  # yaw,
                                                                                                  # ang_vel_x,
                                                                                                  # ang_vel_y,
                                                                                                  # ang_vel_z,
                                                                                                  # rpm0,
                                                                                                  # rpm1,
                                                                                                  # rpm2,
                                                                                                  # rpm3
        #### Note: this is the suggest information to log ##############################
        self.controls = np.zeros((num_drones, 12, duration_sec*self.LOGGING_FREQ_HZ)) #### 12 control targets: pos_x,
                                                                                                             # pos_y,
                                                                                                             # pos_z,
                                                                                                             # vel_x, 
                                                                                                             # vel_y,
                                                                                                             # vel_z,
                                                                                                             # roll,
                                                                                                             # pitch,
                                                                                                             # yaw,
                                                                                                             # ang_vel_x,
                                                                                                             # ang_vel_y,
                                                                                                             # ang_vel_z

    ################################################################################

    def log(self,
            drone: int,
            timestamp,
            state,
            control=np.zeros(12)
            ):
        """Logs entries for a single simulation step, of a single drone.

        Parameters
        ----------
        drone : int
            Id of the drone associated to the log entry.
        timestamp : float
            Timestamp of the log in simulation clock.
        state : ndarray
            (20,)-shaped array of floats containing the drone's state.
        control : ndarray, optional
            (12,)-shaped array of floats containing the drone's control target.

        """
        if drone < 0 or drone >= self.NUM_DRONES or timestamp < 0 or len(state) != 20 or len(control) != 12:
            print("[ERROR] in Logger.log(), invalid data")
        current_counter = int(self.counters[drone])
        #### Add rows to the matrices if a counter exceeds their size
        if current_counter >= self.timestamps.shape[1]:
            self.timestamps = np.concatenate((self.timestamps, np.zeros((self.NUM_DRONES, 1))), axis=1)
            self.states = np.concatenate((self.states, np.zeros((self.NUM_DRONES, 16, 1))), axis=2)
            self.controls = np.concatenate((self.controls, np.zeros((self.NUM_DRONES, 12, 1))), axis=2)
        #### Advance a counter is the matrices have overgrown it ###
        elif not self.PREALLOCATED_ARRAYS and self.timestamps.shape[1] > current_counter:
            current_counter = self.timestamps.shape[1]-1
        #### Log the information and increase the counter ##########
        self.timestamps[drone, current_counter] = timestamp
        #### Re-order the kinematic obs (of most Aviaries) #########
        self.states[drone, :, current_counter] = np.hstack([state[0:3], state[10:13], state[7:10], state[13:20]])
        self.controls[drone, :, current_counter] = control
        self.counters[drone] = current_counter + 1

    ################################################################################

    def save(self):
        """Save the logs to file.
        """
        with open(os.path.join(self.OUTPUT_FOLDER, "save-flight-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+".npy"), 'wb') as out_file:
            np.savez(out_file, timestamps=self.timestamps, states=self.states, controls=self.controls)

    ################################################################################

    def save_as_csv(self,
                    comment: str=""
                    ):
        """Save the logs---on your Desktop---as comma separated values.

        Parameters
        ----------
        comment : str, optional
            Added to the foldername.

        """
        csv_dir = os.path.join(self.OUTPUT_FOLDER, "save-flight-"+comment+"-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir+'/')
        t = np.arange(0, self.timestamps.shape[1]/self.LOGGING_FREQ_HZ, 1/self.LOGGING_FREQ_HZ)
        for i in range(self.NUM_DRONES):
            with open(csv_dir+"/x"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 0, :]])), delimiter=",")
            with open(csv_dir+"/y"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 1, :]])), delimiter=",")
            with open(csv_dir+"/z"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 2, :]])), delimiter=",")
            ####
            with open(csv_dir+"/r"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 6, :]])), delimiter=",")
            with open(csv_dir+"/p"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 7, :]])), delimiter=",")
            with open(csv_dir+"/ya"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 8, :]])), delimiter=",")
            ####
            with open(csv_dir+"/rr"+str(i)+".csv", 'wb') as out_file:
                rdot = np.hstack([0, (self.states[i, 6, 1:] - self.states[i, 6, 0:-1]) * self.LOGGING_FREQ_HZ ])
                np.savetxt(out_file, np.transpose(np.vstack([t, rdot])), delimiter=",")
            with open(csv_dir+"/pr"+str(i)+".csv", 'wb') as out_file:
                pdot = np.hstack([0, (self.states[i, 7, 1:] - self.states[i, 7, 0:-1]) * self.LOGGING_FREQ_HZ ])
                np.savetxt(out_file, np.transpose(np.vstack([t, pdot])), delimiter=",")
            with open(csv_dir+"/yar"+str(i)+".csv", 'wb') as out_file:
                ydot = np.hstack([0, (self.states[i, 8, 1:] - self.states[i, 8, 0:-1]) * self.LOGGING_FREQ_HZ ])
                np.savetxt(out_file, np.transpose(np.vstack([t, ydot])), delimiter=",")
            ###
            with open(csv_dir+"/vx"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 3, :]])), delimiter=",")
            with open(csv_dir+"/vy"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 4, :]])), delimiter=",")
            with open(csv_dir+"/vz"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 5, :]])), delimiter=",")
            ####
            with open(csv_dir+"/wx"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 9, :]])), delimiter=",")
            with open(csv_dir+"/wy"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 10, :]])), delimiter=",")
            with open(csv_dir+"/wz"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 11, :]])), delimiter=",")
            ####
            with open(csv_dir+"/rpm0-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 12, :]])), delimiter=",")
            with open(csv_dir+"/rpm1-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 13, :]])), delimiter=",")
            with open(csv_dir+"/rpm2-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 14, :]])), delimiter=",")
            with open(csv_dir+"/rpm3-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 15, :]])), delimiter=",")
            ####
            with open(csv_dir+"/pwm0-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, (self.states[i, 12, :] - 4070.3) / 0.2685])), delimiter=",")
            with open(csv_dir+"/pwm1-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, (self.states[i, 13, :] - 4070.3) / 0.2685])), delimiter=",")
            with open(csv_dir+"/pwm2-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, (self.states[i, 14, :] - 4070.3) / 0.2685])), delimiter=",")
            with open(csv_dir+"/pwm3-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, (self.states[i, 15, :] - 4070.3) / 0.2685])), delimiter=",")

    ################################################################################
    
    def plot(self, pwm=False):
        """Logs entries for a single simulation step, of a single drone.

        Parameters
        ----------
        pwm : bool, optional
            If True, converts logged RPM into PWM values (for Crazyflies).

        """
        #### Loop over colors and line styles ######################
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) + cycler('linestyle', ['-', '--', ':', '-.'])))
        fig, axs = plt.subplots(10, 2)
        t = np.arange(0, self.timestamps.shape[1]/self.LOGGING_FREQ_HZ, 1/self.LOGGING_FREQ_HZ)

        #### Column ################################################
        col = 0

        #### XYZ ###################################################
        row = 0
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 0, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('x (m)')

        row = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 1, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y (m)')

        row = 2
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 2, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('z (m)')

        #### RPY ###################################################
        row = 3
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 6, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('r (rad)')
        row = 4
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 7, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('p (rad)')
        row = 5
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 8, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y (rad)')

        #### Ang Vel ###############################################
        row = 6
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 9, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wx')
        row = 7
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 10, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wy')
        row = 8
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 11, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wz')

        #### Time ##################################################
        row = 9
        axs[row, col].plot(t, t, label="time")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('time')

        #### Column ################################################
        col = 1

        #### Velocity ##############################################
        row = 0
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 3, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vx (m/s)')
        row = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 4, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vy (m/s)')
        row = 2
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 5, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vz (m/s)')

        #### RPY Rates #############################################
        row = 3
        for j in range(self.NUM_DRONES):
            rdot = np.hstack([0, (self.states[j, 6, 1:] - self.states[j, 6, 0:-1]) * self.LOGGING_FREQ_HZ ])
            axs[row, col].plot(t, rdot, label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('rdot (rad/s)')
        row = 4
        for j in range(self.NUM_DRONES):
            pdot = np.hstack([0, (self.states[j, 7, 1:] - self.states[j, 7, 0:-1]) * self.LOGGING_FREQ_HZ ])
            axs[row, col].plot(t, pdot, label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('pdot (rad/s)')
        row = 5
        for j in range(self.NUM_DRONES):
            ydot = np.hstack([0, (self.states[j, 8, 1:] - self.states[j, 8, 0:-1]) * self.LOGGING_FREQ_HZ ])
            axs[row, col].plot(t, ydot, label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('ydot (rad/s)')

        ### This IF converts RPM into PWM for all drones ###########
        #### except drone_0 (only used in examples/compare.py) #####
        for j in range(self.NUM_DRONES):
            for i in range(12,16):
                if pwm and j > 0:
                    self.states[j, i, :] = (self.states[j, i, :] - 4070.3) / 0.2685

        #### RPMs ##################################################
        row = 6
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 12, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM0')
        else:
            axs[row, col].set_ylabel('RPM0')
        row = 7
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 13, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM1')
        else:
            axs[row, col].set_ylabel('RPM1')
        row = 8
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 14, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM2')
        else:
            axs[row, col].set_ylabel('RPM2')
        row = 9
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 15, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM3')
        else:
            axs[row, col].set_ylabel('RPM3')

        #### Drawing options #######################################
        for i in range (10):
            for j in range (2):
                axs[i, j].grid(True)
                axs[i, j].legend(loc='upper right',
                         frameon=True
                         )
        fig.subplots_adjust(left=0.06,
                            bottom=0.05,
                            right=0.99,
                            top=0.98,
                            wspace=0.15,
                            hspace=0.0
                            )
        if self.COLAB: 
            plt.savefig(os.path.join('results', 'output_figure.png'))
        else:
            plt.show()



================================================
FILE: gym_pybullet_drones/utils/utils.py
================================================
"""General use functions.
"""
import time
import argparse
import numpy as np
from scipy.optimize import nnls

################################################################################

def sync(i, start_time, timestep):
    """Syncs the stepped simulation with the wall-clock.

    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.

    Parameters
    ----------
    i : int
        Current simulation iteration.
    start_time : timestamp
        Timestamp of the simulation start.
    timestep : float
        Desired, wall-clock step of the simulation's rendering.

    """
    if timestep > .04 or i%(int(1/(24*timestep))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (i*timestep):
            time.sleep(timestep*i - elapsed)

################################################################################

def str2bool(val):
    """Converts a string into a boolean.

    Parameters
    ----------
    val : str | bool
        Input value (possibly string) to interpret as boolean.

    Returns
    -------
    bool
        Interpretation of `val` as True or False.

    """
    if isinstance(val, bool):
        return val
    elif val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("[ERROR] in str2bool(), a Boolean value is expected")



================================================
FILE: tests/test_build.py
================================================
def test_imports():
    import gym_pybullet_drones
    import gym_pybullet_drones.control
    import gym_pybullet_drones.envs
    import gym_pybullet_drones.examples
    import gym_pybullet_drones.utils



================================================
FILE: tests/test_examples.py
================================================
def test_pid():
    from gym_pybullet_drones.examples.pid import run
    run(gui=False, plot=False, output_folder='tmp')

def test_pid_velocity():
    from gym_pybullet_drones.examples.pid_velocity import run
    run(gui=False, plot=False, output_folder='tmp')

def test_downwash():
    from gym_pybullet_drones.examples.downwash import run
    run(gui=False, plot=False, output_folder='tmp')

def test_learn():
    from gym_pybullet_drones.examples.learn import run
    run(gui=False, plot=False, output_folder='tmp', local=False)



================================================
FILE: .github/dependabot.yml
================================================
# Please see the documentation for all configuration options:
# https://docs.github.com/code-security/dependabot/dependabot-version-updates/configuration-options-for-the-dependabot.yml-file

version: 2
updates:
  - package-ecosystem: "pip" # See documentation for possible values
    directory: "/" # Location of package manifests
    schedule:
      interval: "weekly"



================================================
FILE: .github/workflows/test.yml
================================================
name: Test Package

on: 
  push:
    branches: [ main ] 
  schedule:
    - cron: '0 3 * * 6'
  workflow_dispatch: {}

permissions:
  contents: read

jobs:
  run_tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest
    - name: Install package
      run: |
        pip install -e .
    - name: Unit tests
      run: |
        python -m pytest tests/
        rm -rf tmp/


