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

Please record one big dataset for training and one small one for validation (model selection via wandb). In our presentation the training set had around 2000 images and 250 for the validation set.

## Train model

To train the model execute the file `train_model.py` and pass as an argument the directory containing the train and validation set.
 
The directory should be designed as follow:

* data/
    * train/
        * imgs/
        * sensor_data.npy
    * val/
        * imgs/
        * sensor_data.npy

We trained our model on a GPU node in [USI HPC cluster](https://intranet.ics.usi.ch/HPC).

## Test model
To test the model please launch Gazebo with the wanted world 
```
roslaunch final_project thymio_gazebo_bringup.launch name:=thymio10 world:=simple
```
and then execute the `avoid_obstacle.launch` file as follow:
```
roslaunch final_project avoid_obstacle.launch
```

## Test Teleoperation with automatic object/pitfall protection

To test the model please launch Gazebo with the wanted world 
```
roslaunch final_project thymio_gazebo_bringup.launch name:=thymio10 world:=pitfalls
```
and then execute the `avoid_obstacle.launch` file as follow:
```
rosrun final_project key_teleop.py key_vel:=/thymio10/cmd_vel
```
## Contacts 

If you have any doubts please contact us at noli.manzoni@usi.ch or micheal.denzler.usi.ch