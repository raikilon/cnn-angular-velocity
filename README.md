# Robotics Final Project
#### Noli Manzoni, Micheal Denzler
Add the `final_project` package to your `your_catkin_workspace`

## Get training data
To get the training data please launch Gazebo with the wanted world
```
roslaunch final_project thymio_gazebo_bringup.launch name:=thymio10 world:=simple
```
and then execute the `random_walk.launch` file as follow:
```
roslaunch final_project random_walk.launch
```
This script will populate the folder `data/imgs` with images from the Thymio (one per second) and once the program is closed with `ctrl+c` it also saves the target data in `sensor_data.npy`
### Run different parts separately

To run the **part 1** (drawing an eight) please replace `compulsory.launch` with `controller_eight.launch` and `world:=wall` with `world:=empty`. Morever, please remove the spawning of the additional Thymio by commenting the include tags in `thymio_gazebo_brinup.launch`.

To run the **part 2** (face the wall) please replace `compulsory.launch` with `controller.launch`.

### Videos and pictures
#### Drawing an eight with two circles
![Drawing an eight with Thymio](eight.jpg)

#### Moving two meters from the wall
[Video on Google Drive](https://drive.google.com/file/d/1VPS1ZS7qb0r1xvPJwJEVVD7TWvGN4RX1/view?usp=sharing)

## Bonus part
To run the **bonus** part of the assignment, first bring up the gazebo (*please remember to copy the `arena.world` inside the folder `launch/world`*):
```
roslaunch thymio_course_skeleton thymio_gazebo_bringup.launch name:=thymio10 world:=arena
```
then execite the `bonus.launch` file as follow:
```
roslaunch thymio_course_skeleton bonus.launch
```
This command will execute the second part of the bonus assignment where three Thymio will wander randomly around the arena. More precisely, once a Thymio hit the wall, it will rotate randomly away from it and then it will move forward.

### Videos
#### One single Thymio

[Video on Google Drive](https://drive.google.com/file/d/1GwoefUJ059tFeFPK2Z-XjVqvOdofRxL5/view?usp=sharing)

#### Two Thymio that get close to each other

[Video on Google Drive](https://drive.google.com/file/d/148APHwss9yPnVXrrVlPQMfp37GTERckd/view?usp=sharing)

# Comments on implementation, problems and doubts

## Compulsory task
Challenging we found the following:
	- part 1: Defininig the turning point while drawing the 8. 
	- part 2/3: To find out if the Thymio's axis is ortogonal to the wall, we checked that the proximities from the left and right camera are the same. If that was the case, we knew that the Thymio was in facing the wall with 90 degrees. Due to the unprecision of the proximity sensors we compared the left and right proximity using np.isclose(). Critical, and difficult, was here to find the right tolerance. This was the case for part 2 when we tried to face the wall orthogonally as well as in part 3 when we tried to face away from the wall orthogonally
	- Part 3: Challenging from an architectural standpoint was to define the different stages of the robot and how to hand-over between the different stage.

## Additional work: 
In part 1 we used Rviz to visualize the 8 the robot was drawing (see attachment eight.jpg).

## Bonus task
In the bonus part, the hardest challenge for us was to handle the launch file.

# Conlusion
All tasks (1-3) plus the bonus have been successfully implemented, the simulations work smoothly. We were able to sovle all our doubts/questions within the team and do not see any puzzling behaviours.

If you have any doubts please contact us at noli.manzoni@usi.ch or micheal.denzler.usi.ch