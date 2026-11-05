@echo off
chcp 65001 > nul
echo Dang chay kiem thu tu dong Pipeline va Checkpoint...
.venv\Scripts\python.exe app.py --test
pause

