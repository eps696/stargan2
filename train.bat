@echo off
rem train dataset 256 8

python src/train.py --data_dir data/%1 --model_dir train/%~n1 ^
--img_size %2 --batch_size %3 ^
--lambda_ds 2 --lambda_cyc 1 --lambda_sty 1 ^
%4 %5 %6 %7 %8 %9
