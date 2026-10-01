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

; The language picker opens first on every interactive install, and its choice also sets Scoobert's language.
!define MUI_LANGDLL_ALWAYSSHOW
!define MUI_LANGDLL_WINDOWTITLE "Scoobert"
!define MUI_LANGDLL_INFO "Choose a language for Scoobert."
!define MUI_LANGDLL_REGISTRY_ROOT HKCU
!define MUI_LANGDLL_REGISTRY_KEY "Software\Scoobert"
!define MUI_LANGDLL_REGISTRY_VALUENAME "InstallerLanguage"

!define MUI_WELCOMEPAGE_TITLE "$(WelcomeTitle)"
!define MUI_WELCOMEPAGE_TEXT "$(WelcomeText)"
!insertmacro MUI_PAGE_WELCOME
!define MUI_DIRECTORYPAGE_TEXT_TOP "$(DirectoryText)"
!define MUI_PAGE_CUSTOMFUNCTION_LEAVE CheckFolder
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_TITLE "$(FinishTitle)"
!define MUI_FINISHPAGE_TEXT "$(FinishText)"
!define MUI_FINISHPAGE_RUN "$INSTDIR\scoobert.exe"
!define MUI_FINISHPAGE_RUN_TEXT "$(FinishRun)"
!insertmacro MUI_PAGE_FINISH

!define MUI_UNCONFIRMPAGE_TEXT_TOP "$(UnConfirmText)"
!insertmacro MUI_UNPAGE_CONFIRM
!define MUI_COMPONENTSPAGE_TEXT_TOP "$(UnComponentsText)"
!insertmacro MUI_UNPAGE_COMPONENTS
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH
!include "languages.nsh"
!insertmacro MUI_RESERVEFILE_LANGDLL

!define UNINSTALL_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\Scoobert"
!define MODELS "$PROFILE\models"

; Refuses a folder the account cannot write to, such as Program Files, before installing anything.
Function CheckFolder
  ClearErrors
  CreateDirectory "$INSTDIR"
  FileOpen $0 "$INSTDIR\.scoobert-write-test" w
  ${If} ${Errors}
    MessageBox MB_OK|MB_ICONEXCLAMATION "$(FolderError)"
    Abort
  ${EndIf}
  FileClose $0
  Delete "$INSTDIR\.scoobert-write-test"
FunctionEnd

; An update started from inside Scoobert installs silently and keeps the language already chosen.
Function .onInit
  ${IfNot} ${Silent}
    !insertmacro MUI_LANGDLL_DISPLAY
  ${EndIf}
FunctionEnd

; An update started from inside Scoobert passes /relaunch, so Scoobert opens again after the silent install.
Function .onInstSuccess
  ${GetParameters} $R0
  ClearErrors
  ${GetOptions} $R0 "/relaunch" $R1
  ${IfNot} ${Errors}
    Exec '"$INSTDIR\scoobert.exe"'
  ${EndIf}
FunctionEnd

Section "Scoobert"
  ; A running copy locks its files, so it is asked to close and ended if it is still open after 10 seconds.
  nsExec::Exec 'taskkill /IM scoobert.exe'
  StrCpy $1 0
  ${DoWhile} ${FileExists} "$INSTDIR\scoobert.exe"
    ClearErrors
    FileOpen $0 "$INSTDIR\scoobert.exe" a
    ${IfNot} ${Errors}
      FileClose $0
      ${Break}
    ${EndIf}
    IntOp $1 $1 + 1
    ${If} $1 = 20
      nsExec::Exec `powershell -NoProfile -Command "Get-Process scoobert -ErrorAction SilentlyContinue | Where-Object { $$_.Path -like '$INSTDIR\*' } | Stop-Process -Force"`
    ${ElseIf} $1 >= 30
      ${Break}
    ${EndIf}
    Sleep 500
  ${Loop}
  ; A model server left behind by a crash or a forced close locks the llama folder.
  nsExec::Exec `powershell -NoProfile -Command "Get-Process llama-server -ErrorAction SilentlyContinue | Where-Object { $$_.Path -like '$INSTDIR\*' } | Stop-Process -Force"`
  SetOutPath "$INSTDIR"
  File /r "${SOURCE}\*.*"
  Delete "$INSTDIR\portable.txt"
  ; Scoobert starts in this language until the user picks another one inside it.
  ${IfNot} ${Silent}
    FileOpen $0 "$INSTDIR\language.txt" w
    FileWrite $0 "$(LangCode)"
    FileClose $0
  ${EndIf}
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
  Delete "$INSTDIR\language.txt"
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
  !insertmacro MUI_DESCRIPTION_TEXT ${UnApp} "$(UnAppDesc)"
  !insertmacro MUI_DESCRIPTION_TEXT ${UnData} "$(UnDataDesc)"
  !insertmacro MUI_DESCRIPTION_TEXT ${UnModels} "$(UnModelsDesc)"
!insertmacro MUI_UNFUNCTION_DESCRIPTION_END

; Shows how much space the models take, and hides the option when there are none.
Function un.onInit
  !insertmacro MUI_UNGETLANGUAGE
  SectionSetText ${UnData} "$(UnDataName)"
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
    SectionSetText ${UnModels} "$(UnModelsName)"
  ${EndIf}
FunctionEnd
