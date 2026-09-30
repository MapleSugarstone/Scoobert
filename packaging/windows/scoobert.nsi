; Per-user installer for Scoobert. Build with scripts/package-windows.ps1, which passes VERSION and SOURCE.
Unicode true
!include "MUI2.nsh"

!ifndef VERSION
  !define VERSION "0.0.0"
!endif
!ifndef SOURCE
  !define SOURCE "..\..\dist\Scoobert"
!endif

Name "Scoobert"
OutFile "..\..\dist\Scoobert-Setup-${VERSION}.exe"
InstallDir "$LOCALAPPDATA\Programs\Scoobert"
InstallDirRegKey HKCU "Software\Scoobert" "InstallDir"
RequestExecutionLevel user
SetCompressor /SOLID lzma

!define MUI_ICON "..\..\assets\icon.ico"
!define MUI_UNICON "..\..\assets\icon.ico"
!define MUI_FINISHPAGE_RUN "$INSTDIR\scoobert.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Open Scoobert"
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

!define UNINSTALL_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\Scoobert"

Section "Scoobert"
  ; A running copy locks its files, so ask it to close first.
  nsExec::Exec 'taskkill /IM scoobert.exe'
  Sleep 1500
  SetOutPath "$INSTDIR"
  File /r "${SOURCE}\*.*"
  WriteUninstaller "$INSTDIR\Uninstall Scoobert.exe"
  CreateShortCut "$SMPROGRAMS\Scoobert.lnk" "$INSTDIR\scoobert.exe"
  CreateShortCut "$DESKTOP\Scoobert.lnk" "$INSTDIR\scoobert.exe"
  WriteRegStr HKCU "Software\Scoobert" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayName" "Scoobert"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "Publisher" "Maple Sugarstone"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayIcon" "$INSTDIR\scoobert.exe"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "UninstallString" '"$INSTDIR\Uninstall Scoobert.exe"'
  WriteRegDWORD HKCU "${UNINSTALL_KEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINSTALL_KEY}" "NoRepair" 1
SectionEnd

; Settings, conversations, and downloaded models stay, so a reinstall picks up where the user left off.
Section "Uninstall"
  nsExec::Exec 'taskkill /IM scoobert.exe'
  nsExec::Exec 'taskkill /IM llama-server.exe'
  Sleep 1500
  RMDir /r "$INSTDIR\llama"
  Delete "$INSTDIR\scoobert.exe"
  Delete "$INSTDIR\LICENSE"
  Delete "$INSTDIR\Uninstall Scoobert.exe"
  RMDir "$INSTDIR"
  Delete "$SMPROGRAMS\Scoobert.lnk"
  Delete "$DESKTOP\Scoobert.lnk"
  DeleteRegKey HKCU "${UNINSTALL_KEY}"
  DeleteRegKey HKCU "Software\Scoobert"
SectionEnd
