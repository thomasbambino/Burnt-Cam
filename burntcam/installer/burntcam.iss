; Inno Setup script for BurntCamSetup.exe.
; Built by .github/workflows/burntcam-installer.yml after installer\build.ps1
; has prepared the app folder. Compile manually with:
;   ISCC.exe /DSourceDir=C:\path\to\build\app /DObsUrl=<OBS installer URL> burntcam.iss

#define MyAppName "Burnt Cam"
#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\build\app"
#endif
#ifndef ObsUrl
  #define ObsUrl ""
#endif
#define VcRedistUrl "https://aka.ms/vs/17/release/vc_redist.x64.exe"

[Setup]
AppId={{8F3B6C2A-5D1E-4F7A-9C3B-2E6D8A1F4B70}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=Burnt Cam
AppComments=3D character webcam filter
DefaultDirName={autopf}\Burnt Cam
DefaultGroupName=Burnt Cam
DisableProgramGroupPage=yes
; Admin once, so OBS and the Visual C++ runtime can be installed in the same go.
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\dist
OutputBaseFilename=BurntCamSetup
SetupIconFile=..\burntcam.ico
UninstallDisplayIcon={app}\burntcam.ico
UninstallDisplayName={#MyAppName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "installobs"; Description: "Install OBS Studio (free) - it provides the ""OBS Virtual Camera"" that Zoom, Discord and Teams use"; GroupDescription: "Virtual camera:"; Check: ObsMissing

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Burnt Cam"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\burntcam.py"""; WorkingDir: "{app}"; IconFilename: "{app}\burntcam.ico"; Comment: "3D character webcam"
Name: "{autodesktop}\Burnt Cam"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\burntcam.py"""; WorkingDir: "{app}"; IconFilename: "{app}\burntcam.ico"; Comment: "3D character webcam"; Tasks: desktopicon

[Run]
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "Installing the Microsoft Visual C++ runtime..."; Flags: waituntilterminated; Check: VcRedistReady
Filename: "{tmp}\obs-installer.exe"; Parameters: "/S"; StatusMsg: "Installing OBS Studio (this can take a minute)..."; Flags: waituntilterminated; Check: ObsReady
Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\burntcam.py"""; WorkingDir: "{app}"; Description: "Start Burnt Cam now"; Flags: nowait postinstall skipifsilent runasoriginaluser

[UninstallDelete]
; Python writes cache files next to the code; remove them too.
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\__pycache__"

[Code]
var
  DownloadPage: TDownloadWizardPage;

function VcRedistNeeded: Boolean;
begin
  Result := not (FileExists(ExpandConstant('{sys}\msvcp140.dll')) and
                 FileExists(ExpandConstant('{sys}\vcruntime140_1.dll')));
end;

function ObsMissing: Boolean;
begin
  Result := not RegKeyExists(HKLM, 'SOFTWARE\OBS Studio') and
            not FileExists(ExpandConstant('{commonpf64}\obs-studio\bin\64bit\obs64.exe'));
end;

function VcRedistReady: Boolean;
begin
  Result := VcRedistNeeded and FileExists(ExpandConstant('{tmp}\vc_redist.x64.exe'));
end;

function ObsReady: Boolean;
begin
  Result := WizardIsTaskSelected('installobs') and FileExists(ExpandConstant('{tmp}\obs-installer.exe'));
end;

function OnDownloadProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
begin
  Result := True;
end;

procedure InitializeWizard;
begin
  DownloadPage := CreateDownloadPage(SetupMessage(msgWizardPreparing), 'Downloading the extras Burnt Cam needs...', @OnDownloadProgress);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Count: Integer;
begin
  Result := True;
  if CurPageID <> wpReady then
    Exit;
  DownloadPage.Clear;
  Count := 0;
  if VcRedistNeeded then begin
    DownloadPage.Add('{#VcRedistUrl}', 'vc_redist.x64.exe', '');
    Count := Count + 1;
  end;
  if WizardIsTaskSelected('installobs') and ('{#ObsUrl}' <> '') then begin
    DownloadPage.Add('{#ObsUrl}', 'obs-installer.exe', '');
    Count := Count + 1;
  end;
  if Count = 0 then
    Exit;
  DownloadPage.Show;
  try
    try
      DownloadPage.Download;
    except
      if DownloadPage.AbortedByUser then
        Result := False
      else
        SuppressibleMsgBox('Some downloads failed: ' + GetExceptionMessage + #13#10#13#10 +
          'Burnt Cam will still be installed. If the virtual camera doesn''t work, install ' +
          'OBS Studio from obsproject.com.', mbError, MB_OK, IDOK);
    end;
  finally
    DownloadPage.Hide;
  end;
end;
