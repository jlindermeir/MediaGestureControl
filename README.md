# MediaGestureControl
Control any media playback simply by using gestures!
The gesture recognition is provided by [20bn-realtimenet](https://github.com/TwentyBN/20bn-realtimenet) 
and [player_do](https://pypi.org/project/playerdo/) is used to communicate with the media player.

Tested on Ubuntu 20.04.1 for Spotify and Chrome playing YouTube.

## Installation
The script assumes that [20bn-realtimenet](https://github.com/TwentyBN/20bn-realtimenet) is cloned and correctly 
installed in the folder beside this repository. The structure should look like this:
```
<your_project_folder>
├── MediaGestureControl
├── 20bn-realtimenet
```
You can also use a different project structure, simply alter the symlinks to the `realtimenet` python package and the
`resources/` directory.

Additionally the [player_do](https://pypi.org/project/playerdo/) python package as well as the `dbus` python bindings 
need to be installed.

## Usage
Simply start your favorite media player and run
```
python run.py
```
Try out a few gestures. Laying a finger on your lips should pause the playback, while giving a thumbs up should 
start it again.
