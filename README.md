# Build and install

Create a workspace
```
mkdir -p ~/topoly_ws/src
```
Clone the repository:
```
cd ~/topoly_ws/src
```
```
git clone https://github.com/juandpenan/topology_nav_ros2.git
```
Import third parties:
```
vcs import  < topology_nav_ros2/thirdparty.repos
```
Install dependencies:
```
pip install -r topology_nav_ros2/requirements.txt
```
```
cd ~/topoly_ws
```
```
rosdep update && rosdep install --from-paths src --ignore-src -y --skip-keys 'libpcl-dev'
```
```
colcon build --symlink-install --cmake-args -DBUILD_TESTING=OFF
```
or
```
colcon build --symlink-install
```
# How to run:

## Execute mapping node:
```
cd ~/topoly_ws
```
```
source ~/topology_ws/install/setup.bash
```
one one terminal launch simulation:
```
ros2 launch topological_mapping map.launch.py 
```
open another terminal and launch rviz2:
```
rviz2
```
- with rviz open use the set initial pose button to determine the initial pose.
- move the robot where you want to extract semantic features of the map.

## Execute localization:
```
cd ~/topoly_ws
```
```
source ~/topology_ws/install/setup.bash
```
one one terminal launch simulation:
```
ros2 launch topological_localization demo_mh_amcl.launch.py 
```
