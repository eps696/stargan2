@echo off

python src/test.py --model models/%1 --source _in/%2 --out_dir _out/%~n1 ^
%3 %4 %5 %6 %7 %8 %9
