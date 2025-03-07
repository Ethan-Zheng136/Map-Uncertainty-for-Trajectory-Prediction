# export PYTHONPATH="${PYTHONPATH}:/MapBEVPrediction/MapTRv2_modified"
# python tools/test.py \
#     /MapBEVPrediction/MapTRv2_modified/projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py \
#     /MapBEVPrediction/MapTRv2_modified/work_dirs/maptrv2_nusc_r50_24ep/YOURCHECKPOINT.pth \
#     --eval chamfer \
#     --bev_path /path_to_save_bev_features

# python tools/test.py \
#     /MapBEVPrediction/MapTRv2_modified/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_w_centerline.py \
#     /MapBEVPrediction/MapTRv2_modified/work_dirs/maptrv2_nusc_r50_24ep_w_centerline/YOUR_CHECKPOINT.pth \
#     --eval chamfer \
#     --bev_path /path_to_save_bev_features
export PYTHONPATH="${PYTHONPATH}:/MapBEVPrediction/MapTRv2_modified"
export CUDA_VISIBLE_DEVICES=5
# python tools/test.py \
#     /DATA_EDS2/zhangbr2407/MapBEVPrediction/MapTRv2_modified/projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py \
#     /DATA_EDS2/zhangbr2407/MapBEVPrediction/MapTRv2_modified/ckpts/maptrv2.pth \
#     --eval chamfer \
#     --bev_path /DATA_EDS2/zhangbr2407/MapBEVPrediction/bev_val_2


python tools/test.py \
    /DATA_EDS2/zhangbr2407/MapBEVPrediction/MapTRv2_modified/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_w_centerline.py \
    /DATA_EDS2/zhangbr2407/MapBEVPrediction/HiVT_modified/ckpts/maptrv2_cent.pth \
    --eval chamfer \
    --bev_path /DATA_EDS2/zhangbr2407/MapBEVPrediction/bev_train_2c