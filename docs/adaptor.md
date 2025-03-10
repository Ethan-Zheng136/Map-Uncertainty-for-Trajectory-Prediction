## Merging Map Data with Trajectory Data

### Necessary Files
- `mapping_results.pickle` from mapping evaluation
- `bev_features` stored in previous `/path_to_save_bev_features`
- Trajdata pickle files: `traj_scene_frame_{full_train, full_val, mini_train, mini_val}.pkl`
- Ground truth files: `gt_{full_train, full_val, mini_val}.pickle`

You can download the above `Trajdata pickle files` and `Ground truth files` files from [here](https://drive.google.com/drive/folders/17kqpilI4dP6ZY7XFnWArPbMSlpDq8ErM?usp=drive_link)

### Data Merging Process

First create a new directory to store the above files, and a new directory to store the merged map & trajectory data

```
mkdir adaptor_files trj_data
cd trj_data
mkdir maptr maptrv2 maptrv2_cent stream
cd ..
```

#### Folder Structure
```
DelvingUncPrediction
├── nuscenes/
├── processed/
├── adaptor_files/
│   ├── traj_scene_frame_{full_train, full_val, mini_train, mini_val}.pkl
│   ├── gt_{full_train, full_val, mini_val}.pickle
│   ├── mapping_results.pickle
├── trj_data/
│   ├── maptr/
│   ├── maptrv2/
│   ├── maptrv2_cent/
│   ├── stream/
```

#### Merging Map Data and Trajectory Data

Run the script `adaptor.py` in `MapBEVPrediction/adaptor/` to merge mapping data and trajectory data. 

```
cd adaptor

# To merge MapTRv2 centerline, add in an additional argument --centerline
python adaptor.py \
  --version mini \                                       # [trainval, mini]
  --split mini_val \                                     # [train, train_val, val, mini_train, mini_val]
  --map_model MapTR \                                    # [MapTR, StreamMapNet]
  --dataroot ../nuscenes \
  --index_file ../adaptor_files/traj_scene_frame_mini_val.pkl \
  --map_file ../adaptor_files/mapping_results.pickle \
  --gt_map_file ../adaptor_files/gt_mini_val.pickle \
  --save_path ../trj_data/maptr                          # trj_data directory, [maptr, maptrv2, maptrv2_cent, stream]

# For visualization
python adaptor_vis.py --data_path ../trj_data/maptr                        
```

After running the above script, **folder structure** should look like this:
```
DelvingUncPrediction
├── nuscenes/
├── processed/
├── adaptor_files/
├── trj_data/
|   ├── maptr/
│   |   ├── mini_val/
│   |   |   ├── data/
│   |   |   |   ├── scene-{scene_id}.pkl
│   |   ├── train/
│   |   ├── val/
│   ├── maptrv2/
│   ├── maptrv2_cent/
│   ├── stream/
```

#### Merging in BEV Features

To merge in the intermediate BEV features produced during map estimation, run the following:

```
python adaptor_bev.py \
  --scene_folder /path_to_scene_folder \
  --bev_folder /path_to_save_bev_features \
  --map_model {MapTR, StreamMapNet}
```

**Note** that this merging process needs to be conducted at least twice per mapping method. Once on train split, once on val split. You need mapping results from evaluating on both the train and val set for each mapping method. So in total, the mapping evaluation scripts should be run 8 times (2 times 4 mapping methods). If you only want to examine one of the mapping method, then 2 times are enough. 

