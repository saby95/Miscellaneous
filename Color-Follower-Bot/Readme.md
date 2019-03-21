# Color-Follower Bot

The aim of the project is to navigate a robot on the arena shown in the picture. The starting point of the robot is the big black square in the middle. Starting from there, the robot should travel to the red, blue and green circles, while avoiding the small black squares and return back to the big black square after completion

This was implemented as part of a contest and as part of the contest each coloraturas is given values such as 1500, 1000 and 500 respectively. The Bot should collect the number of points given on spot, in the minimum time. We won the contest completing it in the minimum time.

The python file monitors the positioning of the bot in the arena and communicates with the MSP430 micro controller on the robot serially using the 'Code-for-energia.ino' to navigate the robot.