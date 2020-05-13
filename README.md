# Robotics Final Project

#### Noli Manzoni, Micheal Denzler

If you want to have more information about our implementaiton please look at project description in the pdf file.

## How to use

First add the `final_project` package to your `your_catkin_workspace`

### Get training data
To get the training data please launch Gazebo with the wanted world (simple or pitfalls)
```
roslaunch final_project thymio_gazebo_bringup.launch name:=thymio10 world:=simple
```
and then execute the `random_walk.launch` file as follow:
```
roslaunch final_project random_walk.launch
```
This script will populate the folder `data/imgs` with images from the Thymio (one per second) and once the program is closed with `ctrl+c` (better to do this when the Thymio is moving forward) it also saves the target data in `sensor_data.npy` (when using the save & flag system it will save also ` pitfall_flags.npy` and `object_flags.npy` ).

Please record one big dataset for training and one small one for validation (model selection via wandb). In our presentation the training set had around 2000 images and 250 for the validation set.

#### Collection systems

Longer ranges: random_walk.launch

Save & flag: random_walk_pitfalls.launch

Teleoperation: random_walk_teleop.launch

### Train model

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

If you use the save and flag system, please change the dataset import in the top of `train_model.py`and add the additional file to the data directory.

Already trained model can be found at the following [link](https://mega.nz/folder/FhQjVQhB#WGYx3LL-L5fwcznx5PM3tw) where  `pitfalls.tar` is a model trained with the save and flag system on the pitfalls map and `obstacles.tar` is the model trained with long ranges on the simple map.

### Test model
To test the model put the `.tar` file in the model directory and then  launch Gazebo with the wanted world (simple or pitfalls)
```bash
roslaunch final_project thymio_gazebo_bringup.launch name:=thymio10 world:=simple
```
or the test world

```
roslaunch final_project thymio_test_gazebo_bringup.launch
```

and then execute the `avoid_obstacle.launch` file with the chosen model (pitfalls or obstacles) as follow (X is 10 for normal world and 11 to 16 for the test world):

```
roslaunch final_project avoid_obstacle.launch name:=thymioX model:=pitfalls
```

#### Teleoperation

To test the model please launch Gazebo with the wanted world 

```
roslaunch final_project thymio_gazebo_bringup.launch name:=thymio10 world:=pitfalls
```
and then execute the `avoid_obstacle.launch` file as follow:
```
roslaunch final_project teleoperate.launch
```
### Contacts 

If you have any doubts please contact us at noli.manzoni@usi.ch or micheal.denzler.usi.ch