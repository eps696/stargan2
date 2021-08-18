@echo off
echo .. %1 .. %2
if "%3"=="" goto help

:seq_src
if exist _in\%~n2\* goto process
echo .. making source sequence
mkdir _in\%~n2
ffmpeg -y -v warning -i _in\%2 -f image2 -start_number 0 _in\%~n2\%%06d.jpg

:process
echo .. processing
python src/process.py --model models/%1 --source _in/%~n2 --refs %3 ^
%4 %5 %6 %7 %8 %9

ffmpeg -y -v warning -i _out\%~n2\%%06d.jpg _out\%~n2-%~n1.mp4

goto end

:help
echo Usage: process model src refs
echo  e.g.: process afhq-256-3-100.ckpt test.avi afhq_dir
echo  e.g.: process afhq-256-3-100.ckpt test.avi 0-0-1-1-2
:end


