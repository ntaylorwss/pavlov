# Release History
## 0.2.0
### Docker changes
- Make testing a part of the build procedure.
- Even though we're no longer rendering to screen, env.render requires pyglet, so must pass the screen. Doing so now.
- Always use most up to date version of Pavlov when building.
- Build all images (GPU, CPU, dev) simultaneously in bootstrap script.

### Pavlov changes
- Apparently I forgot to import top level modules in init. Did that now. Woopsies.
- Added one end-to-end test, and unit tests for pipeline functions as well as value schedules.
- Rewrote documentation to comply with Numpy style guide.
- Generally, enabled use of package outside of associated Docker container.
- Removed use of formatted strings to comply with Python <3.6.
- Removed use of keep_running file, instead use custom context manager for clean termination of indefinite runs.
- Created automated check for model NaNs during training.
- Timestamped video files output by agents.

## 0.1.3
### Changes
- Changed from GIF output to exporting to MP4 for watching environment renderings.

## 0.1.2
Actual successful(-ish) initial release.

## 0.1.1
Initial release, but I flubbed it. First time using PyPi and I did dumb things. Had to skip this version.
