# Monte Carlo Localization
* This is a Python implementation of the Monte Carlo Localization algorithm for robot movement data obtained by a turtle-bot within a university classroom (CSE_668.bag).
* Work done as part of CSE 668 - Advanced Robotics taught by Nils Napp at the University at Buffalo.
* landmarks.csv contains landmark location data.
* Uses rosbag to read the BAG file, hence ROS needs to be installed. Tested on ROS Kinetic.
* Run the code:
```
python monte-carlo.py --steps 20 --num_particles 200
```
* steps - Number of localization steps for the Monte Carlo algorithm.
* num_particles - Number of particles to initialize the particle filter with.
