XSOCK=/tmp/.X11-unix
XAUTH=$HOME/.Xauthority
XIMAGE=/home/hydrapc/Documents/share

VOLUMES="--volume=$XSOCK:$XSOCK:rw
         --volume=$XAUTH:$XAUTH:rw
         --volume=$XIMAGE:/home/mobilitylab:rw"

IMAGE=liangkailiu/rt-bev-uniad:v0.3

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

