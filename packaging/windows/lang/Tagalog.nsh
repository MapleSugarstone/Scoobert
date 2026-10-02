;Language: Tagalog (1124)
;Translation by Scoobert

!insertmacro LANGFILE "Tagalog" = "Tagalog" "Tagalog"

!ifdef MUI_WELCOMEPAGE
  ${LangFileString} MUI_TEXT_WELCOME_INFO_TITLE "Maligayang pagdating sa Setup ng $(^NameDA)"
  ${LangFileString} MUI_TEXT_WELCOME_INFO_TEXT "Gagabayan ka ng Setup sa pag-install ng $(^NameDA).$\r$\n$\r$\nInirerekomendang isara mo ang lahat ng iba pang app bago simulan ang Setup. Magagawa nitong i-update ang mga kaugnay na system file nang hindi kinakailangang i-restart ang computer mo.$\r$\n$\r$\n$_CLICK"
!endif

!ifdef MUI_UNWELCOMEPAGE
  ${LangFileString} MUI_UNTEXT_WELCOME_INFO_TITLE "Maligayang pagdating sa Pag-uninstall ng $(^NameDA)"
  ${LangFileString} MUI_UNTEXT_WELCOME_INFO_TEXT "Gagabayan ka ng Setup sa pag-uninstall ng $(^NameDA).$\r$\n$\r$\nBago simulan ang pag-uninstall, tiyaking hindi tumatakbo ang $(^NameDA).$\r$\n$\r$\n$_CLICK"
!endif

!ifdef MUI_LICENSEPAGE
  ${LangFileString} MUI_TEXT_LICENSE_TITLE "Kasunduan sa Lisensya"
  ${LangFileString} MUI_TEXT_LICENSE_SUBTITLE "Pakirepaso ang mga tuntunin ng lisensya bago i-install ang $(^NameDA)."
  ${LangFileString} MUI_INNERTEXT_LICENSE_BOTTOM "Kung tinatanggap mo ang mga tuntunin ng kasunduan, i-click ang Sumasang-ayon Ako para magpatuloy. Kailangan mong tanggapin ang kasunduan para ma-install ang $(^NameDA)."
  ${LangFileString} MUI_INNERTEXT_LICENSE_BOTTOM_CHECKBOX "Kung tinatanggap mo ang mga tuntunin ng kasunduan, lagyan ng tsek ang check box sa ibaba. Kailangan mong tanggapin ang kasunduan para ma-install ang $(^NameDA). $_CLICK"
  ${LangFileString} MUI_INNERTEXT_LICENSE_BOTTOM_RADIOBUTTONS "Kung tinatanggap mo ang mga tuntunin ng kasunduan, piliin ang unang opsyon sa ibaba. Kailangan mong tanggapin ang kasunduan para ma-install ang $(^NameDA). $_CLICK"
!endif

!ifdef MUI_UNLICENSEPAGE
  ${LangFileString} MUI_UNTEXT_LICENSE_TITLE "Kasunduan sa Lisensya"
  ${LangFileString} MUI_UNTEXT_LICENSE_SUBTITLE "Pakirepaso ang mga tuntunin ng lisensya bago i-uninstall ang $(^NameDA)."
  ${LangFileString} MUI_UNINNERTEXT_LICENSE_BOTTOM "Kung tinatanggap mo ang mga tuntunin ng kasunduan, i-click ang Sumasang-ayon Ako para magpatuloy. Kailangan mong tanggapin ang kasunduan para ma-uninstall ang $(^NameDA)."
  ${LangFileString} MUI_UNINNERTEXT_LICENSE_BOTTOM_CHECKBOX "Kung tinatanggap mo ang mga tuntunin ng kasunduan, lagyan ng tsek ang check box sa ibaba. Kailangan mong tanggapin ang kasunduan para ma-uninstall ang $(^NameDA). $_CLICK"
  ${LangFileString} MUI_UNINNERTEXT_LICENSE_BOTTOM_RADIOBUTTONS "Kung tinatanggap mo ang mga tuntunin ng kasunduan, piliin ang unang opsyon sa ibaba. Kailangan mong tanggapin ang kasunduan para ma-uninstall ang $(^NameDA). $_CLICK"
!endif

!ifdef MUI_LICENSEPAGE | MUI_UNLICENSEPAGE
  ${LangFileString} MUI_INNERTEXT_LICENSE_TOP "Pindutin ang Page Down para makita ang natitirang bahagi ng kasunduan."
!endif

!ifdef MUI_COMPONENTSPAGE
  ${LangFileString} MUI_TEXT_COMPONENTS_TITLE "Pumili ng mga Component"
  ${LangFileString} MUI_TEXT_COMPONENTS_SUBTITLE "Piliin kung aling mga feature ng $(^NameDA) ang gusto mong i-install."
!endif

!ifdef MUI_UNCOMPONENTSPAGE
  ${LangFileString} MUI_UNTEXT_COMPONENTS_TITLE "Pumili ng mga Component"
  ${LangFileString} MUI_UNTEXT_COMPONENTS_SUBTITLE "Piliin kung aling mga feature ng $(^NameDA) ang gusto mong i-uninstall."
!endif

!ifdef MUI_COMPONENTSPAGE | MUI_UNCOMPONENTSPAGE
  ${LangFileString} MUI_INNERTEXT_COMPONENTS_DESCRIPTION_TITLE "Paglalarawan"
  !ifndef NSIS_CONFIG_COMPONENTPAGE_ALTERNATIVE
    ${LangFileString} MUI_INNERTEXT_COMPONENTS_DESCRIPTION_INFO "Ilagay ang mouse sa isang component para makita ang paglalarawan nito."
  !else
    ${LangFileString} MUI_INNERTEXT_COMPONENTS_DESCRIPTION_INFO "Pumili ng component para makita ang paglalarawan nito."
  !endif
!endif

!ifdef MUI_DIRECTORYPAGE
  ${LangFileString} MUI_TEXT_DIRECTORY_TITLE "Pumili ng Lokasyon ng Pag-install"
  ${LangFileString} MUI_TEXT_DIRECTORY_SUBTITLE "Piliin ang folder kung saan i-install ang $(^NameDA)."
!endif

!ifdef MUI_UNDIRECTORYPAGE
  ${LangFileString} MUI_UNTEXT_DIRECTORY_TITLE "Pumili ng Lokasyon ng Pag-uninstall"
  ${LangFileString} MUI_UNTEXT_DIRECTORY_SUBTITLE "Piliin ang folder kung saan i-uninstall ang $(^NameDA)."
!endif

!ifdef MUI_INSTFILESPAGE
  ${LangFileString} MUI_TEXT_INSTALLING_TITLE "Ini-install"
  ${LangFileString} MUI_TEXT_INSTALLING_SUBTITLE "Pakihintay habang ini-install ang $(^NameDA)."
  ${LangFileString} MUI_TEXT_FINISH_TITLE "Tapos na ang Pag-install"
  ${LangFileString} MUI_TEXT_FINISH_SUBTITLE "Matagumpay na natapos ang Setup."
  ${LangFileString} MUI_TEXT_ABORT_TITLE "Naihinto ang Pag-install"
  ${LangFileString} MUI_TEXT_ABORT_SUBTITLE "Hindi matagumpay na natapos ang Setup."
!endif

!ifdef MUI_UNINSTFILESPAGE
  ${LangFileString} MUI_UNTEXT_UNINSTALLING_TITLE "Ina-uninstall"
  ${LangFileString} MUI_UNTEXT_UNINSTALLING_SUBTITLE "Pakihintay habang ina-uninstall ang $(^NameDA)."
  ${LangFileString} MUI_UNTEXT_FINISH_TITLE "Tapos na ang Pag-uninstall"
  ${LangFileString} MUI_UNTEXT_FINISH_SUBTITLE "Matagumpay na natapos ang pag-uninstall."
  ${LangFileString} MUI_UNTEXT_ABORT_TITLE "Naihinto ang Pag-uninstall"
  ${LangFileString} MUI_UNTEXT_ABORT_SUBTITLE "Hindi matagumpay na natapos ang pag-uninstall."
!endif

!ifdef MUI_FINISHPAGE
  ${LangFileString} MUI_TEXT_FINISH_INFO_TITLE "Tinatapos ang Setup ng $(^NameDA)"
  ${LangFileString} MUI_TEXT_FINISH_INFO_TEXT "Na-install na ang $(^NameDA) sa computer mo.$\r$\n$\r$\nI-click ang Tapusin para isara ang Setup."
  ${LangFileString} MUI_TEXT_FINISH_INFO_REBOOT "Kailangang i-restart ang computer mo para makumpleto ang pag-install ng $(^NameDA). Gusto mo bang mag-restart ngayon?"
!endif

!ifdef MUI_UNFINISHPAGE
  ${LangFileString} MUI_UNTEXT_FINISH_INFO_TITLE "Tinatapos ang Pag-uninstall ng $(^NameDA)"
  ${LangFileString} MUI_UNTEXT_FINISH_INFO_TEXT "Na-uninstall na ang $(^NameDA) sa computer mo.$\r$\n$\r$\nI-click ang Tapusin para isara ang Setup."
  ${LangFileString} MUI_UNTEXT_FINISH_INFO_REBOOT "Kailangang i-restart ang computer mo para makumpleto ang pag-uninstall ng $(^NameDA). Gusto mo bang mag-restart ngayon?"
!endif

!ifdef MUI_FINISHPAGE | MUI_UNFINISHPAGE
  ${LangFileString} MUI_TEXT_FINISH_REBOOTNOW "Mag-restart ngayon"
  ${LangFileString} MUI_TEXT_FINISH_REBOOTLATER "Gusto kong mag-restart nang mano-mano mamaya"
  ${LangFileString} MUI_TEXT_FINISH_RUN "&Patakbuhin ang $(^NameDA)"
  ${LangFileString} MUI_TEXT_FINISH_SHOWREADME "&Ipakita ang Readme"
  ${LangFileString} MUI_BUTTONTEXT_FINISH "&Tapusin"  
!endif

!ifdef MUI_STARTMENUPAGE
  ${LangFileString} MUI_TEXT_STARTMENU_TITLE "Pumili ng Folder sa Start Menu"
  ${LangFileString} MUI_TEXT_STARTMENU_SUBTITLE "Pumili ng folder sa Start Menu para sa mga shortcut ng $(^NameDA)."
  ${LangFileString} MUI_INNERTEXT_STARTMENU_TOP "Piliin ang folder sa Start Menu kung saan mo gustong gawin ang mga shortcut ng program. Maaari ka ring maglagay ng pangalan para gumawa ng bagong folder."
  ${LangFileString} MUI_INNERTEXT_STARTMENU_CHECKBOX "Huwag gumawa ng mga shortcut"
!endif

!ifdef MUI_UNCONFIRMPAGE
  ${LangFileString} MUI_UNTEXT_CONFIRM_TITLE "I-uninstall ang $(^NameDA)"
  ${LangFileString} MUI_UNTEXT_CONFIRM_SUBTITLE "Alisin ang $(^NameDA) sa computer mo."
!endif

!ifdef MUI_ABORTWARNING
  ${LangFileString} MUI_TEXT_ABORTWARNING "Sigurado ka bang gusto mong lumabas sa Setup ng $(^Name)?"
!endif

!ifdef MUI_UNABORTWARNING
  ${LangFileString} MUI_UNTEXT_ABORTWARNING "Sigurado ka bang gusto mong lumabas sa Pag-uninstall ng $(^Name)?"
!endif

!ifdef MULTIUSER_INSTALLMODEPAGE
  ${LangFileString} MULTIUSER_TEXT_INSTALLMODE_TITLE "Pumili ng mga User"
  ${LangFileString} MULTIUSER_TEXT_INSTALLMODE_SUBTITLE "Piliin kung para sa aling mga user mo gustong i-install ang $(^NameDA)."
  ${LangFileString} MULTIUSER_INNERTEXT_INSTALLMODE_TOP "Piliin kung gusto mong i-install ang $(^NameDA) para sa sarili mo lang o para sa lahat ng user ng computer na ito. $(^ClickNext)"
  ${LangFileString} MULTIUSER_INNERTEXT_INSTALLMODE_ALLUSERS "I-install para sa sinumang gumagamit ng computer na ito"
  ${LangFileString} MULTIUSER_INNERTEXT_INSTALLMODE_CURRENTUSER "I-install para sa akin lang"
!endif
