export PYTHONPATH="${PYTHONPATH}:/MapBEVPrediction/MapTR_modified"
export CUDA_VISIBLE_DEVICES=1
/DATA_EDS2/zhangbr2407/.conda/envs/map1/bin/python tools/test.py \
    /DATA_EDS2/zhangbr2407/MapBEVPrediction/MapTR_modified/projects/configs/maptr/maptr_tiny_r50_24e.py \
    /DATA_EDS2/zhangbr2407/MapBEVPrediction/HiVT_modified/ckpts/maptr.pth \
    --eval chamfer \
    --bev_path /DATA_EDS2/zhangbr2407/MapBEVPrediction/bev_train_2
