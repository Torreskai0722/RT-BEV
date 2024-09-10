XSOCK=/tmp/.X11-unix
XAUTH=$HOME/.Xauthority

VOLUMES="--volume=$XSOCK:$XSOCK:rw
         --volume=$XAUTH:$XAUTH:rw"

IMAGE=liangkailiu/rt-bev-uniad:v1.1

echo "Launching $IMAGE"

sudo docker run \
    -it \
    --gpus all \
    --shm-size=30g \
    $VOLUMES \
    --env="XAUTHORITY=${XAUTH}" \
    --env="DISPLAY=${DISPLAY}" \
    --privileged \
    --net=host \
    $RUNTIME \
    $IMAGE \
    /bin/bash

