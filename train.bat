@echo off

echo Start Training Batch
python multi_label_classifier.py --dir %1 --name .\%2 --model VGG16 --pretrain --sum_epoch 64 --save_epoch_freq 64 --input_size 200 --load_size 225 --batch_size 32 --top_k (1,) --shuffle --image_ncols 5

echo Start Test
python multi_label_classifier.py --dir %1 --name .\%2 --model VGG16 --mode test --checkpoint_name epoch_64_snapshot.pth --pretrain --input_size 200 --load_size 225
