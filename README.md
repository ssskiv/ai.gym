this is template for creating ROS2 Humble container with Nvidia GPU support (with automatic integrated mesa GPU mode) and Webots preinstalled. Building requires stable Internet connection.
<h1>How to use?</h1>

1. Change `CONTAINER_NAME` in `Makefile`

2. Print `make` in terminal in folder with `Makefile` and wait...

<h1>Host machine requirements</h1>

Ubuntu 22.04 with Nvidia 570(you can change this in Dockerfile and Makefile) for Nvidia GPU support.

Ubuntu 22.04 with any video drivers for Mesa mode.
