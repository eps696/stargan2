@echo off
echo .. %1 .. %2
if "%3"=="" goto help

:seq_src
if exist %~dp2\%~n2\* goto process
echo .. making source sequence
mkdir %~dp2\%~n2
ffmpeg -y -v warning -i %2 -f image2 -start_number 0 %~dp2\%~n2\%%06d.jpg

:process
echo .. processing
python src/process.py --model %1 --source %~dp2/%~n2 --refs %3 --out_dir _out/%~n2-%~n1-%~n3 ^
%4 %5 %6 %7 %8 %9

ffmpeg -y -v warning -i _out\%~n2-%~n1-%~n3\%%06d.jpg _out\%~n2-%~n1-%~n3.mp4

goto end

:help
echo Usage: proc model src refs
echo  e.g.: proc afhq-256-3-100.ckpt pu.avi afhq_dir
echo  e.g.: proc afhq-256-3-100.ckpt pu.avi 0-0-1-1-2
:end


