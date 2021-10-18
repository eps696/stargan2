@echo off
if "%4"=="" goto help
echo .. %1 .. %2

if exist %3 goto file
set src=size
set recur=1
goto process
:file
set src=source
set recur=0.4
set dirname=%~n3

:process
echo .. processing
python src/process.py --model models/%1 --refs %2 --%src% %3 --frames %4 --out_dir _out/%~n1-%~n3 -r %recur% ^
%5 %6 %7 %8 %9

ffmpeg -y -v warning -i _out\%~n1-%~n3\%dirname%\%%06d.jpg _out\%~n1-%~n3.mp4

goto end

:help
echo Usage: recurs model refs size frames
echo  e.g.: recurs afhq-256-3-100.ckpt 0-1-2 1280-720 100-25
echo    or: recurs model refs sourcefile frames
echo  e.g.: recurs afhq-256-3-100.ckpt 0-1-2 _in/mapping.jpg 100-25
:end
