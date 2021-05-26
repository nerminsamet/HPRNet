# train
python src/main.py --task landmark --exp_id coco_wholebody_hg --dataset coco_body --arch hourglass --gpus 0,1,2,3

# test
python src/test.py --task landmark --exp_id coco_wholebody_hg --dataset coco_body --arch hourglass --batch_size 1 --keep_res --K 25 --resume --load_model ./models/coco_wholebody_hourglass.pth

# flip test
python src/test.py --task landmark --exp_id coco_wholebody_hg --dataset coco_body --arch hourglass --batch_size 1 --keep_res --K 25 --resume --load_model ./models/coco_wholebody_hourglass.pth --flip_test
