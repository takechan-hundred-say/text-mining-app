@echo off
chcp 65001 > nul
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
"$WshShell = New-Object -ComObject WScript.Shell; ^
$Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\計量テキスト分析.lnk'); ^
$Shortcut.TargetPath = '%~dp0start.bat'; ^
$Shortcut.WorkingDirectory = '%~dp0'; ^
$Shortcut.IconLocation = '%~dp0textview_view_search_find.ico'; ^
$Shortcut.Description = '計量テキスト分析'; ^
$Shortcut.Save();"

set PYTHON=python
if exist "python\python.exe" set PYTHON=python\python.exe

echo 必要なパッケージを確認・インストールしています...
%PYTHON% -m pip install -r requirements.txt

echo アプリを起動しています...
%PYTHON% -m streamlit run main.py

pause