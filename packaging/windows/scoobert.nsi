; Per-user installer for Scoobert. Build with scripts/package-windows.ps1, which passes VERSION and SOURCE.
Unicode true
!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "LogicLib.nsh"

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
BrandingText "Scoobert ${VERSION}"

!define MUI_ICON "..\..\assets\icon.ico"
!define MUI_UNICON "..\..\assets\icon.ico"
!define MUI_ABORTWARNING

!define MUI_WELCOMEPAGE_TITLE "Install Scoobert"
!define MUI_WELCOMEPAGE_TEXT "Scoobert is a coding assistant that runs AI models on your own computer.$\r$\n$\r$\nIt installs for your Windows account only, so you don't need an administrator password.$\r$\n$\r$\nSelect Next to continue."
!insertmacro MUI_PAGE_WELCOME
!define MUI_DIRECTORYPAGE_TEXT_TOP "Scoobert will be installed in the folder below. To put it somewhere else, such as on another drive, select Browse."
!define MUI_PAGE_CUSTOMFUNCTION_LEAVE CheckFolder
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_TITLE "Scoobert is installed"
!define MUI_FINISHPAGE_TEXT "You can open Scoobert from its shortcut on the desktop or in the Start menu.$\r$\n$\r$\nTo remove it later, open Settings in Scoobert and select Uninstall, or use Installed apps in Windows Settings."
!define MUI_FINISHPAGE_RUN "$INSTDIR\scoobert.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Open Scoobert now"
!insertmacro MUI_PAGE_FINISH

!define MUI_UNCONFIRMPAGE_TEXT_TOP "Scoobert will be removed from this folder. Your projects and their notes are never touched."
!insertmacro MUI_UNPAGE_CONFIRM
!define MUI_COMPONENTSPAGE_TEXT_TOP "Choose what else to remove. Anything you leave checked off stays, so a reinstall continues where you left off."
!insertmacro MUI_UNPAGE_COMPONENTS
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH
!insertmacro MUI_LANGUAGE "English"

!define UNINSTALL_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\Scoobert"
!define MODELS "$PROFILE\models"

; Refuses a folder the account cannot write to, such as Program Files, before installing anything.
Function CheckFolder
  ClearErrors
  CreateDirectory "$INSTDIR"
  FileOpen $0 "$INSTDIR\.scoobert-write-test" w
  ${If} ${Errors}
    MessageBox MB_OK|MB_ICONEXCLAMATION "Windows does not let Scoobert install in that folder without an administrator. Pick a folder inside your own user folder, or keep the one Scoobert suggested."
    Abort
  ${EndIf}
  FileClose $0
  Delete "$INSTDIR\.scoobert-write-test"
FunctionEnd

Section "Scoobert"
  ; A running copy locks its files, so it is asked to close first.
  nsExec::Exec 'taskkill /IM scoobert.exe'
  Sleep 1500
  SetOutPath "$INSTDIR"
  File /r "${SOURCE}\*.*"
  Delete "$INSTDIR\portable.txt"
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
  WriteRegStr HKCU "${UNINSTALL_KEY}" "QuietUninstallString" '"$INSTDIR\Uninstall Scoobert.exe" /S'
  WriteRegDWORD HKCU "${UNINSTALL_KEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINSTALL_KEY}" "NoRepair" 1
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  WriteRegDWORD HKCU "${UNINSTALL_KEY}" "EstimatedSize" $0
SectionEnd

Section "un.Scoobert" UnApp
  SectionIn RO
  nsExec::Exec 'taskkill /IM scoobert.exe'
  ; Only the model server that belongs to this copy of Scoobert.
  nsExec::Exec `powershell -NoProfile -Command "Get-Process llama-server -ErrorAction SilentlyContinue | Where-Object { $$_.Path -like '$INSTDIR\*' } | Stop-Process -Force"`
  Sleep 1500
  RMDir /r "$INSTDIR\llama"
  Delete "$INSTDIR\scoobert.exe"
  Delete "$INSTDIR\LICENSE"
  Delete "$INSTDIR\Uninstall Scoobert.exe"
  RMDir "$INSTDIR"
  Delete "$SMPROGRAMS\Scoobert.lnk"
  Delete "$DESKTOP\Scoobert.lnk"
  ; Saved prompt caches are only a speed-up, so they always go.
  RMDir /r "$LOCALAPPDATA\Scoobert"
  DeleteRegKey HKCU "${UNINSTALL_KEY}"
  DeleteRegKey HKCU "Software\Scoobert"
SectionEnd

Section /o "un.Settings and conversations" UnData
  RMDir /r "$APPDATA\Scoobert"
SectionEnd

Section /o "un.Downloaded models" UnModels
  RMDir /r "${MODELS}\Qwen3.5-9B-Q4_K_M"
  RMDir /r "${MODELS}\Qwen3.8-27B-UD-IQ4_XS"
  RMDir /r "${MODELS}\Qwen3.8-27B-UD-Q6_K"
  RMDir /r "${MODELS}\Qwen3.8-Flash-Next-UD-IQ3_XXS"
  ; The folder itself goes only when nothing else is in it.
  RMDir "${MODELS}"
SectionEnd

!insertmacro MUI_UNFUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${UnApp} "The Scoobert program, its shortcuts, and its saved prompt caches."
  !insertmacro MUI_DESCRIPTION_TEXT ${UnData} "Your project list, settings, and past conversations. Project notes stay in your projects."
  !insertmacro MUI_DESCRIPTION_TEXT ${UnModels} "The Qwen models Scoobert downloaded to your models folder. Downloading them again takes a while."
!insertmacro MUI_UNFUNCTION_DESCRIPTION_END

; Shows how much space the models take, and hides the option when there are none.
Function un.onInit
  StrCpy $R0 0
  ${ForEach} $R1 1 4 + 1
    ${If} $R1 == 1
      StrCpy $R2 "${MODELS}\Qwen3.5-9B-Q4_K_M"
    ${ElseIf} $R1 == 2
      StrCpy $R2 "${MODELS}\Qwen3.8-27B-UD-IQ4_XS"
    ${ElseIf} $R1 == 3
      StrCpy $R2 "${MODELS}\Qwen3.8-27B-UD-Q6_K"
    ${Else}
      StrCpy $R2 "${MODELS}\Qwen3.8-Flash-Next-UD-IQ3_XXS"
    ${EndIf}
    ${If} ${FileExists} "$R2\*.*"
      ${GetSize} "$R2" "/S=0M" $0 $1 $2
      IntOp $R0 $R0 + $0
    ${EndIf}
  ${Next}
  ${If} $R0 == 0
    SectionSetText ${UnModels} ""
  ${Else}
    IntOp $R3 $R0 / 1024
    SectionSetText ${UnModels} "Downloaded models (about $R3 GB)"
  ${EndIf}
FunctionEnd
