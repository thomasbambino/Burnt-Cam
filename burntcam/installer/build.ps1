# Builds the folder that BurntCamSetup.exe installs: the app code, the face
# model and a private Python with every package already installed.
# Runs on a Windows machine (GitHub Actions); see .github/workflows/burntcam-installer.yml.
param([string]$Out = "build\app")

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Src = Resolve-Path (Join-Path $PSScriptRoot '..')
$PyVersion = '3.12.10'

if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Out = Resolve-Path $Out
$PyDir = Join-Path $Out 'python'
$Py = Join-Path $PyDir 'python.exe'
$Tmp = Join-Path $env:RUNNER_TEMP 'burntcam-build'
if (-not $env:RUNNER_TEMP) { $Tmp = Join-Path $env:TEMP 'burntcam-build' }
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null

Write-Host "== Python $PyVersion (embeddable)"
$zip = Join-Path $Tmp 'python.zip'
Invoke-WebRequest "https://www.python.org/ftp/python/$PyVersion/python-$PyVersion-embed-amd64.zip" -OutFile $zip -UseBasicParsing
Expand-Archive $zip -DestinationPath $PyDir -Force
$pth = Get-ChildItem -Path $PyDir -Filter 'python*._pth' | Select-Object -First 1
(Get-Content $pth.FullName) -replace '^#\s*import site', 'import site' | Set-Content $pth.FullName -Encoding ASCII

Write-Host "== pip"
$getpip = Join-Path $Tmp 'get-pip.py'
Invoke-WebRequest 'https://bootstrap.pypa.io/get-pip.py' -OutFile $getpip -UseBasicParsing
& $Py $getpip --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw 'get-pip failed' }

Write-Host "== packages"
& $Py -m pip install --no-cache-dir --no-warn-script-location -r (Join-Path $Src 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }

Write-Host "== app files"
Copy-Item (Join-Path $Src '*.py') $Out
Copy-Item (Join-Path $Src 'burntcam.ico') $Out
Copy-Item (Join-Path $Src 'README.md') $Out
Copy-Item (Join-Path $Src 'textures') $Out -Recurse
New-Item -ItemType Directory -Force -Path (Join-Path $Out 'models') | Out-Null
Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task' `
    -OutFile (Join-Path $Out 'models\face_landmarker.task') -UseBasicParsing
# Tells burntcam.py to keep settings/backgrounds/logs in %LOCALAPPDATA%\BurntCam.
Set-Content -Path (Join-Path $Out 'installed.txt') -Value 'Installed by BurntCamSetup.exe. Your settings and backgrounds are in %LOCALAPPDATA%\BurntCam.'

Write-Host "== smoke test"
$test = @"
import os, sys
sys.path.insert(0, r'$Out')
import numpy as np, cv2, moderngl, pyvirtualcam, cv2_enumerate_cameras
import skins, renderer, tracker
t = tracker.FaceTracker(os.path.join(r'$Out', 'models', 'face_landmarker.task'))
assert t.process(np.zeros((480, 640, 3), np.uint8), 0) is None
t.close()
print('smoke test OK - OpenCV', cv2.__version__, '- cameras:', len(cv2_enumerate_cameras.enumerate_cameras(cv2.CAP_DSHOW)))
"@
& $Py -c $test
if ($LASTEXITCODE -ne 0) { throw 'smoke test failed' }

Write-Host "== tidy"
Get-ChildItem -Path $Out -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force
$size = (Get-ChildItem -Path $Out -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ("App folder ready: {0:N0} MB in {1}" -f $size, $Out)
