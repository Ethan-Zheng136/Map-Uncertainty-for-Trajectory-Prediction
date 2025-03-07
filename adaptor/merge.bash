/DATA_EDS2/zhangbr2407/.conda/envs/map11/bin/python adaptor.py \
  --version trainval \
  --split val \
  --map_model MapTR\
  --dataroot ../nuscenes \
  --index_file /DATA_EDS2/zhangbr2407/MapBEVPrediction/adaptor_files/traj_scene_frame_full_val.pkl \
  --map_file /DATA_EDS2/zhangbr2407/MapBEVPrediction/MapTRv2_modified/maptrv2_cent_val.pickle \
  --gt_map_file /DATA_EDS2/zhangbr2407/MapBEVPrediction/adaptor_files/gt_full_val.pickle \
  --save_path ../trj_data2/maptrv2_cent\
  --centerline

