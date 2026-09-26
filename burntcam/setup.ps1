# Burnt Cam setup - run by "Burnt Cam.bat" every time it starts.
#
# The first time, it installs everything Burnt Cam needs; after that it only
# checks and exits in a second or two:
#   1. A private copy of Python in %LOCALAPPDATA%\BurntCam (doesn't touch any
#      other Python on the PC, needs no admin rights).
#   2. The Microsoft Visual C++ runtime, if missing (needed by the face tracker).
#   3. OBS Studio, which provides the "OBS Virtual Camera" that video apps see.
#   4. Burnt Cam's Python packages (re-run when requirements.txt changes).
#   5. A "Burnt Cam" shortcut on the desktop.
# Exit code 0 = ready to run.

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is very slow with the progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$App        = Split-Path -Parent $MyInvocation.MyCommand.Path
$Home_      = Join-Path $env:LOCALAPPDATA 'BurntCam'
$PyDir      = Join-Path $Home_ 'python'
$Py         = Join-Path $PyDir 'python.exe'
$PyVersion  = '3.12.10'
$PyZipUrl   = "https://www.python.org/ftp/python/$PyVersion/python-$PyVersion-embed-amd64.zip"
$GetPipUrl  = 'https://bootstrap.pypa.io/get-pip.py'
$VcRedistUrl = 'https://aka.ms/vs/17/release/vc_redist.x64.exe'

function Say($msg)  { Write-Host "[Burnt Cam] $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "[Burnt Cam] $msg" -ForegroundColor Yellow }

function Download($url, $dest) {
    for ($i = 1; $i -le 3; $i++) {
        try {
            Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
            return
        } catch {
            if ($i -eq 3) { throw "Could not download $url - check your internet connection. ($($_.Exception.Message))" }
            Start-Sleep -Seconds (2 * $i)
        }
    }
}

# Run an installer that needs admin rights (Windows shows its "Allow changes?" prompt).
function Install-Elevated($exe, $arguments) {
    try {
        $p = Start-Process -FilePath $exe -ArgumentList $arguments -Verb RunAs -Wait -PassThru
        return $p.ExitCode
    } catch {
        return -1   # the user said No to the prompt
    }
}

New-Item -ItemType Directory -Force -Path $Home_ | Out-Null
$Tmp = Join-Path $Home_ 'downloads'
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null

# ------------------------------------------------------------------ 1. Python
if (-not (Test-Path $Py)) {
    Say "First-time setup. This downloads about 300 MB and takes a few minutes."
    Say "Downloading Python $PyVersion..."
    $zip = Join-Path $Tmp 'python.zip'
    Download $PyZipUrl $zip
    if (Test-Path $PyDir) { Remove-Item -Recurse -Force $PyDir }
    Expand-Archive -Path $zip -DestinationPath $PyDir -Force
    Remove-Item $zip

    # The "embeddable" Python ignores site-packages until we allow it.
    $pth = Get-ChildItem -Path $PyDir -Filter 'python*._pth' | Select-Object -First 1
    (Get-Content $pth.FullName) -replace '^#\s*import site', 'import site' | Set-Content $pth.FullName -Encoding ASCII

}

# pip (checked separately so an interrupted first run repairs itself)
& $Py -m pip --version *> $null
if ($LASTEXITCODE -ne 0) {
    Say "Installing pip..."
    $getpip = Join-Path $Tmp 'get-pip.py'
    Download $GetPipUrl $getpip
    & $Py $getpip --no-warn-script-location
    if ($LASTEXITCODE -ne 0) { throw "Installing pip failed." }
    Remove-Item $getpip
    Remove-Item (Join-Path $Home_ 'requirements.installed') -ErrorAction SilentlyContinue
}

# ---------------------------------------------------- 2. Visual C++ runtime
$sys32 = Join-Path $env:WINDIR 'System32'
if (-not ((Test-Path (Join-Path $sys32 'msvcp140.dll')) -and (Test-Path (Join-Path $sys32 'vcruntime140_1.dll')))) {
    Say "Installing the Microsoft Visual C++ runtime (Windows will ask for permission)..."
    $vc = Join-Path $Tmp 'vc_redist.x64.exe'
    Download $VcRedistUrl $vc
    $code = Install-Elevated $vc '/install /quiet /norestart'
    if ($code -ne 0 -and $code -ne 3010 -and $code -ne 1638) {
        Warn "The Visual C++ runtime wasn't installed (code $code). Burnt Cam may not start."
    }
    Remove-Item $vc -ErrorAction SilentlyContinue
}

# --------------------------------------------------------------- 3. OBS Studio
function Test-Obs {
    if (Test-Path 'HKLM:\SOFTWARE\OBS Studio') { return $true }
    foreach ($root in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if ($root -and (Test-Path (Join-Path $root 'obs-studio\bin\64bit\obs64.exe'))) { return $true }
    }
    return $false
}

$skipObs = Join-Path $Home_ 'skip-obs'
if (-not (Test-Obs) -and -not (Test-Path $skipObs)) {
    Say "Installing OBS Studio - it provides the virtual camera that Zoom/Discord/Teams see."
    Say "Windows will ask for permission. (You never need to open OBS itself.)"
    $installed = $false

    # Preferred: winget (built into Windows 10/11), which always gets the current version.
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        & winget install --exact --id OBSProject.OBSStudio --silent --accept-package-agreements --accept-source-agreements
        $installed = Test-Obs
    }
    # Fallback: the latest installer straight from OBS's GitHub releases.
    if (-not $installed) {
        try {
            $rel = Invoke-RestMethod -Uri 'https://api.github.com/repos/obsproject/obs-studio/releases/latest' -UseBasicParsing
            $asset = $rel.assets | Where-Object { $_.name -match 'Installer\.exe$' -and $_.name -notmatch 'arm64' } |
                     Sort-Object { if ($_.name -match 'x64') { 0 } else { 1 } } | Select-Object -First 1
            if ($asset) {
                $obs = Join-Path $Tmp $asset.name
                Say "Downloading $($asset.name)..."
                Download $asset.browser_download_url $obs
                Install-Elevated $obs '/S' | Out-Null
                Remove-Item $obs -ErrorAction SilentlyContinue
                $installed = Test-Obs
            }
        } catch {
            Warn "Couldn't download OBS: $($_.Exception.Message)"
        }
    }
    if (-not $installed) {
        Warn "OBS Studio isn't installed, so Burnt Cam will run in preview-only mode."
        Warn "Install it from https://obsproject.com, or delete $skipObs to let Burnt Cam try again."
        New-Item -ItemType File -Force -Path $skipObs | Out-Null
    }
}

# ------------------------------------------------------------- 4. Packages
$req = Join-Path $App 'requirements.txt'
$stamp = Join-Path $Home_ 'requirements.installed'
$want = (Get-FileHash $req -Algorithm SHA256).Hash
$have = if (Test-Path $stamp) { (Get-Content $stamp -Raw).Trim() } else { '' }
if ($want -ne $have) {
    Say "Installing Burnt Cam's packages (face tracking, 3D, virtual camera)..."
    & $Py -m pip install --disable-pip-version-check --no-warn-script-location --upgrade -r $req
    if ($LASTEXITCODE -ne 0) { throw "Installing packages failed - see the messages above." }
    Set-Content -Path $stamp -Value $want
}

# ------------------------------------------------------ 5. Desktop shortcut
$shortcutMade = Join-Path $Home_ 'shortcut-made'
if (-not (Test-Path $shortcutMade)) {
    try {
        $desktop = [Environment]::GetFolderPath('Desktop')
        $shell = New-Object -ComObject WScript.Shell
        $lnk = $shell.CreateShortcut((Join-Path $desktop 'Burnt Cam.lnk'))
        $lnk.TargetPath = Join-Path $App 'Burnt Cam.bat'
        $lnk.WorkingDirectory = $App
        $lnk.Description = 'Burnt Cam - 3D character webcam'
        $lnk.IconLocation = "$env:WINDIR\System32\shell32.dll,204"
        $lnk.Save()
        Say "Added a 'Burnt Cam' shortcut to your desktop."
    } catch {
        Warn "Couldn't create a desktop shortcut: $($_.Exception.Message)"
    }
    New-Item -ItemType File -Force -Path $shortcutMade | Out-Null
}

exit 0
