# train
python src/main.py --task landmark --exp_id coco_wholebody_dla --dataset coco_body --arch dla_34 --gpus 0,1,2,3

# test
python src/test.py --task landmark --exp_id coco_wholebody_dla --dataset coco_body --arch dla_34 --batch_size 1 --keep_res --K 25 --resume --load_model ./models/coco_wholebody_dla34.pth

# flip test
python src/test.py --task landmark --exp_id coco_wholebody_dla --dataset coco_body --arch dla_34 --batch_size 1 --keep_res --K 25 --resume --load_model ./models/coco_wholebody_dla34.pth --flip_test

