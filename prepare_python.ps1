$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonDir = Join-Path $ProjectRoot "python"

Write-Host "=== 組み込み Python 環境のセットアップ ===" -ForegroundColor Cyan

$pyVersion = $null

# Try to detect existing Python version
try {
    $pyVersion = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>$null
} catch {}

if ($pyVersion) {
    Write-Host "現在のPython: $pyVersion" -ForegroundColor Green
} else {
    Write-Host "Python が見つかりません。Python 3.13.7 をダウンロードします..." -ForegroundColor Yellow
    $pyVersion = "3.13.7"
}

$embedUrl = "https://www.python.org/ftp/python/$pyVersion/python-${pyVersion}-embed-amd64.zip"
$embedZip = Join-Path $env:TEMP "python-embed.zip"

Write-Host "ダウンロード: $embedUrl" -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri $embedUrl -OutFile $embedZip -ErrorAction Stop
    Write-Host "ダウンロード完了" -ForegroundColor Green
} catch {
    Write-Host "バージョン $pyVersion の embed が見つかりません。最新版を確認します..." -ForegroundColor Yellow
    $baseUrl = "https://www.python.org/ftp/python/"
    $listing = Invoke-WebRequest -Uri $baseUrl -UseBasicParsing
    $links = [Regex]::Matches($listing.Content, 'href="(\d+\.\d+\.\d+)/"')
    $versions = $links.Groups | Where-Object { $_.Value -match '^\d+\.\d+\.\d+$' } | ForEach-Object { $_.Value }
    $latest313 = $versions | Where-Object { $_ -like "3.13.*" } | Sort-Object -Descending | Select-Object -First 1
    if (-not $latest313) {
        throw "Python 3.13.x の embed パッケージが見つかりませんでした。"
    }
    $embedUrl = "https://www.python.org/ftp/python/$latest313/python-${latest313}-embed-amd64.zip"
    $embedZip = Join-Path $env:TEMP "python-embed.zip"
    Write-Host "再試行: $embedUrl" -ForegroundColor Yellow
    Invoke-WebRequest -Uri $embedUrl -OutFile $embedZip
    $pyVersion = $latest313
    Write-Host "ダウンロード完了（$pyVersion）" -ForegroundColor Green
}

if (Test-Path $PythonDir) {
    Remove-Item -Path $PythonDir -Recurse -Force
}
New-Item -ItemType Directory -Path $PythonDir -Force | Out-Null
Expand-Archive -Path $embedZip -DestinationPath $PythonDir -Force
Write-Host "展開完了" -ForegroundColor Green

$pthFile = Get-ChildItem -Path $PythonDir -Filter "python*._pth" | Select-Object -First 1
if ($pthFile) {
    $content = Get-Content $pthFile.FullName
    $content = $content -replace '#import site', 'import site'
    Set-Content -Path $pthFile.FullName -Value $content
    Write-Host "python._pth を編集（import site 有効化）" -ForegroundColor Green
}

$pythonExe = Join-Path $PythonDir "python.exe"

$getPip = Join-Path $env:TEMP "get-pip.py"
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPip
& $pythonExe $getPip --no-warn-script-location 2>&1 | Out-Null
Write-Host "pip インストール完了" -ForegroundColor Green

$requirements = Join-Path $ProjectRoot "requirements.txt"
if (Test-Path $requirements) {
    Write-Host "パッケージをインストール中..." -ForegroundColor Yellow
    & $pythonExe -m pip install -r $requirements --quiet 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "全パッケージのインストール完了" -ForegroundColor Green
    } else {
        Write-Host "一部のパッケージでエラーが発生しました。手動で確認してください。" -ForegroundColor Red
    }
}

$scriptsDir = Join-Path $PythonDir "Scripts"
if (Test-Path $scriptsDir) {
    Get-ChildItem -Path $scriptsDir -Filter "*.exe" | ForEach-Object {
        Copy-Item $_.FullName $PythonDir -Force
    }
}

Write-Host ""
Write-Host "=== セットアップ完了 ===" -ForegroundColor Cyan
Write-Host "組み込み Python: $PythonDir" -ForegroundColor White
Write-Host "バージョン: $pyVersion" -ForegroundColor White
