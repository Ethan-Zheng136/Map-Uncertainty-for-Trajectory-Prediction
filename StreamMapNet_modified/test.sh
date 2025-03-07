export PYTHONPATH="${PYTHONPATH}:/MapBEVPrediction/StreamMapNet_modified"
export CUDA_VISIBLE_DEVICES=1
python tools/test.py \
    plugin/configs/nusc_newsplit_480_60x30_24e.py \
    /DATA_EDS2/zhangbr2407/MapBEVPrediction/HiVT_modified/ckpts/stream.pth \
    --eval \
    --bev_path /DATA_EDS2/zhangbr2407/MapBEVPrediction/bev_stream_train
