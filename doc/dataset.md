# nuScenes V1.0 Dataset Preparation

You can prepare either the **nuScenes V1.0 mini** dataset or the **nuScenes V1.0 full** dataset for use with UniAD. Follow the instructions for the version you wish to use, ensuring the correct structure for each dataset.

---

### nuScenes V1.0 Mini (Pre-installed in Docker Image)

The **nuScenes V1.0 mini dataset** is a smaller version of the full dataset, useful for testing and development. Download it from [HERE](https://www.nuscenes.org/download) and follow the steps below to prepare the data.

#### **Download nuScenes V1.0 mini**
```shell
cd
mkdir -p nuscenes-mini
# Download nuScenes V1.0 mini dataset directly to (or soft link to) RT-BEV/nuscenes-mini/
```

#### **Prepare UniAD data info**

*Option 1: Generate data infos yourself (for v1.0 mini dataset):*
```shell
cd nuscenes-mini
./tools/uniad_create_data.sh # change the path accordingly
# This will generate nuscenes_infos_temporal_{train,val}.pkl for mini dataset
```

*Option 2: Use off-the-shelf data infos (for full v1.0 dataset):*
```shell
cd nuscenes-mini
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/nuscenes_infos_temporal_train.pkl  # train_infos (mini version)
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/nuscenes_infos_temporal_val.pkl  # val_infos (mini version)
```

#### **Prepare Motion Anchors**
```shell
cd nuscenes-mini
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/motion_anchor_infos_mode6.pkl
```

---

### nuScenes V1.0 Full

Download the **nuScenes V1.0 full dataset**, CAN bus, and map(v1.3) extensions from [HERE](https://www.nuscenes.org/download), and follow the steps below to prepare the data.

#### **Download nuScenes, CAN bus, and Map extensions**
```shell
mkdir -p nuscenes-full
# Download nuScenes V1.0 full dataset directly to (or soft link to) RT-BEV/nuscenes-full/
# Download CAN_bus and Map(v1.3) extensions directly to (or soft link to) RT-BEV/nuscenes-full/
```

#### **Prepare UniAD data info**

*Option 1: Use off-the-shelf data infos (recommended):*
```shell
cd nuscenes-full
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/nuscenes_infos_temporal_train.pkl  # train_infos
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/nuscenes_infos_temporal_val.pkl  # val_infos
```

*Option 2: Generate data infos yourself:*
```shell
cd nuscenes-full
./tools/uniad_create_data.sh
# This will generate nuscenes_infos_temporal_{train,val}.pkl
```

#### **Prepare Motion Anchors**
```shell
cd nuscenes-full
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/motion_anchor_infos_mode6.pkl
```

---

### The Overall Structure

Please ensure that the directory structure is as follows:

#### For **nuScenes V1.0 mini**, the structure should look like this:

```
RT-BEV
├── projects/
├── tools/
├── configs/
├── ckpts/
│   ├── bevformer_r101_dcn_24ep.pth
│   ├── uniad_base_track_map.pth
nuscenes-mini/
├── samples/
├── sweeps/
├── nuscenes_infos_temporal_train.pkl
├── nuscenes_infos_temporal_val.pkl
├── motion_anchor_infos_mode6.pkl
```

#### For **nuScenes V1.0 full**, the structure should look like this:

```
RT-BEV
├── projects/
├── tools/
├── configs/
├── ckpts/
│   ├── bevformer_r101_dcn_24ep.pth
│   ├── uniad_base_track_map.pth
nuscenes-full/
├── can_bus/
├── maps/
├── samples/
├── sweeps/
├── nuscenes_infos_temporal_train.pkl
├── nuscenes_infos_temporal_val.pkl
├── motion_anchor_infos_mode6.pkl
```
