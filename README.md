# COMP755
Real Time Human Pose Detection and Reproduction on Animated Character

# COMP781 Robotics Course Project

Spring 2018, Project title: Balancing Complexity and Optimality of Rapidly-exploring Randomized Tree

## Getting Started

This project implement an RRT variant that combines RRT-Connect and RRTstar.

### Prerequisites

Ubuntu 14.04 is recommended.

Required libraries:

* [OMPL](https://ompl.kavrakilab.org/) - the Open Motion Planning Library (v1.3.2)
* [FCL](https://github.com/flexible-collision-library/fcl/releases) - Flexible Collision Library (v0.3.4)
* [Boost](https://www.boost.org/) - v1.66.0
* [CCD](https://github.com/danfis/libccd)
* [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page) - v3.3.4
* [Catkin](https://github.com/ros/catkin) - which is usually installed with ROS, ROS indigo is recommended for Baxter. (Catkin is not required when doing pure simulation)


### Installing

After all reqired libraries are built, we can start building this project using CMake:

```
git clone git@github.com:mengyu-fu/COMP781.git
cd COMP781
mkdir build
cd build
cmake ..
```

After that, we can use

```
ccmake .
```

and modify the paths for the libraries. Finally, we can do:

```
make
```

## Running the tests

To run the test, we first need to make it:


```
make run-test
```

After that, a test is ready to run. We can use "--help" to check available input parameters:

```
./src/apps/run-test --help
```

Now, try to run with your own parameters.

## Authors

* **Mengyu Fu** - *PhD Student* - [Homepage](http://mengyu.web.unc.edu/)

* **Hao Jiang** - *PhD Student* - [Homepage](http://cs.unc.edu/~haojiang/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Thank you for your interest! The project is still in development...
