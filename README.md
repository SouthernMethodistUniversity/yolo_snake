# yolo_snake

## Get Started

Clone the repo and set up the python environment.

`conda env create -f environment.yml`

## To Play

Open 2 terminal sessions.

In one session launch `snake.py`, in the other launch `detector.py`. If you launch the detector first, it will become difficult to launch snake because of the keyboard input.

Detector.py is designed to use a YOLO model to detect arrows and "press" the associated arrow key. As long as the snake game is the active window on your machine, these inputs will control the snake and allow you to play the game.
