DBoW2
=====

----------------------
Building instructions
----------------------

Required tools:
* CMake to build the code
* git
* C/C++ compiler (gcc >= 4.6 or visual studio or clang)


###  Dependencies

DBoW2 depends on

- DLib (https://github.com/dorian3d/DLib)
- OpenCV
- Boost::dynamic_bitset


On a recent Ubuntu-like distribution (eg 14.04), you may want to try to run::

    $ sudo apt-get install libboost-dev


### Building the dependencies


#### OpenCV 2.4.9
Create a `build` directory where to build the library. It also advisable to set an non-system install directory, so that it will be easier to set up the environment later:
```bash
mkdir build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=`pwd`/install
make install -j n
```
where `n` is the number of threads to use for the compilation.



#### DLib
You can refer to the original documentation [here](https://github.com/dorian3d/DLib).

In short you can just run
```bash
mkdir build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=`pwd`/install -DOpenCV_DIR=<your path to install/share/OpenCV/ dir of opencv>
make install -j n
```

###  Building:

```bash
$ mkdir build && cd build && cmake .. -DOpenCV_DIR=<your path to install/share/OpenCV/ dir of opencv> -DDLIB_ROOT_DIR=<the path you installed DLib>
```