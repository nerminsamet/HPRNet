
# train
python src/main.py --task multi_pose --exp_id coco_wholebody_dla_baseline --dataset coco_body --arch dla_34 --gpus 0,1,2,3

# test
python src/test.py --task multi_pose --exp_id coco_wholebody_dla_baseline --dataset coco_body --arch dla_34 --batch_size 1 --keep_res --resume --load_model ./models/coco_wholebody_baseline.pth








