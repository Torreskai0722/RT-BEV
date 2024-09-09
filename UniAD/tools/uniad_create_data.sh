
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/create_data.py nuscenes --root-path /home/hydrapc/Downloads/v1.0-mini \
       --out-dir /home/hydrapc/Downloads/v1.0-mini \
       --extra-tag nuscenes \
       --version v1.0-mini \
       --canbus /media/hydrapc/hdd-drive3/UniAD-data/nuscenes \