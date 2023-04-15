# snake_robot
snake robot modeling by solidworks and export stl to urdf. Control code base on python and using the Serpernoid control function 

# Startup 
Step1: git clone the repo to your src directory file and catkin_make the workspace 

Step2: Ctrl + T Open terminal (Open 2 terminal for roslaunch )

Step3: (source devel/setup.bash to activate your workspace )

    cd fyp_ws
    
    source devel/setup.bash
    
    roslaunch snake2_gazebo snake2_world.launch
    
    roslaunch snake2_control snake2_control.launch
    
    rosrun snake2_control snake2.py

** the files （snake2_gazebo.launch) stored are used to load the gazebo world and some worlds parameter such as gravity, simulation time accel, obstacles, etc...
** (snake2_control.launch) stored the file to load the ros gazebo controller.
**  snake2_control snake2
  


