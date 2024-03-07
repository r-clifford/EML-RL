#!/bin/bash
docker run -it -v $(pwd):/f1tenth --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
--device=/dev/kfd --device=/dev/dri --group-add video \
-v /tmp/.X11-unix/:/tmp/.X11-unix/ \
-e DISPLAY=:0 \
--ipc=host --shm-size 8G rocm/pytorch:latest
