; Setup program of Starfish Site for Windows, made by Inno Setup 6.
;
; The classic wizard of Inno Setup, with the tall picture on its first and
; last pages, matches the Windows software of 2000 to 2010. Setup installs for
; the current user only and needs no admin rights. build.py compiles it with
;
;   ISCC /DAppVersion=0.1.0 /DSourceDir=<PyInstaller folder> /DArtDir=<rendered pictures>
;        /DOutputDir=<folder> /DLicenseFile=<LICENSE> starfish-site.iss

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "dist\Starfish Site"
#endif
#ifndef ArtDir
  #define ArtDir "art"
#endif
#ifndef OutputDir
  #define OutputDir "dist"
#endif
#ifndef LicenseFile
  #define LicenseFile "..\..\LICENSE"
#endif

[Setup]
AppId={{FA5E8B49-83F1-47DF-B85A-335FE8C28594}
AppName=Starfish Site
AppVersion={#AppVersion}
AppVerName=Starfish Site {#AppVersion}
AppPublisher=Starfish-FL
AppPublisherURL=https://github.com/denoslab/starfish-fl
AppSupportURL=https://denoslab.com/babelbrain-fl-docs/
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\Starfish Site
DefaultGroupName=Starfish Site
DisableProgramGroupPage=yes
DisableWelcomePage=no
PrivilegesRequired=lowest
WizardStyle=classic
WizardImageFile={#ArtDir}\wizard-100.bmp,{#ArtDir}\wizard-150.bmp,{#ArtDir}\wizard-200.bmp
WizardSmallImageFile={#ArtDir}\header-100.bmp,{#ArtDir}\header-150.bmp,{#ArtDir}\header-200.bmp
SetupIconFile={#ArtDir}\starfish-site.ico
UninstallDisplayIcon={app}\Starfish Site.exe
UninstallDisplayName=Starfish Site
LicenseFile={#LicenseFile}
OutputDir={#OutputDir}
OutputBaseFilename=StarfishSite-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
; A running app holds this mutex, and Setup asks to close it first
AppMutex=StarfishSiteApp
CloseApplications=yes
ChangesAssociations=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
WelcomeLabel2=This will install [name/ver] on your computer.%n%nStarfish Site lets this computer help train a shared BabelBrain model together with other hospitals. The files of your cases stay on this computer.%n%nIt is recommended that you close all other applications before continuing.

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Starfish Site"; Filename: "{app}\Starfish Site.exe"
Name: "{group}\Uninstall Starfish Site"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Starfish Site"; Filename: "{app}\Starfish Site.exe"; Tasks: desktopicon

[Registry]
; A double-click on an invitation file opens it in Starfish Site
Root: HKA; Subkey: "Software\Classes\.starfish-invitation"; ValueType: string; ValueName: ""; ValueData: "StarfishSite.Invitation"; Flags: uninsdeletevalue uninsdeletekeyifempty
Root: HKA; Subkey: "Software\Classes\StarfishSite.Invitation"; ValueType: string; ValueName: ""; ValueData: "Starfish Invitation"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\StarfishSite.Invitation\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\Starfish Site.exe,0"
Root: HKA; Subkey: "Software\Classes\StarfishSite.Invitation\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\Starfish Site.exe"" ""%1"""
; The app sets this value when a person chooses to start Starfish at log on, and it goes with the app
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: none; ValueName: "Starfish Site"; Flags: uninsdeletevalue dontcreatekey

[Run]
Filename: "{app}\Starfish Site.exe"; Description: "&Start Starfish Site"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{app}\starfish-site.exe"; Parameters: "stop --now --wait 30"; Flags: runhidden; RunOnceId: "StopSite"

[Code]
// The site's own process holds files of the app open, so it stops before an update replaces them
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
begin
  if FileExists(ExpandConstant('{app}\starfish-site.exe')) then
    Exec(ExpandConstant('{app}\starfish-site.exe'), 'stop --now --wait 30', '', SW_HIDE,
         ewWaitUntilTerminated, ResultCode);
  Result := '';
end;

// The site folder keeps the settings, the logs and the files of each round. Removing the app keeps
// it unless the person asks to delete it, and a silent uninstall keeps it.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  SiteFolder: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    SiteFolder := ExpandConstant('{%USERPROFILE}\.starfish-site');
    if DirExists(SiteFolder) then
      if SuppressibleMsgBox('Starfish Site kept this computer''s settings and logs in ' + SiteFolder +
                            '.' + #13#10#13#10 + 'Do you want to delete them as well?',
                            mbConfirmation, MB_YESNO, IDNO) = IDYES then
        DelTree(SiteFolder, True, True, True);
  end;
end;
