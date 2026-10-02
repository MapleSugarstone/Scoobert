;Language: Sorani (1170)
;Translation for Scoobert
;Structure based on Arabic.nsh

!insertmacro LANGFILE "Sorani" = "کوردیی سۆرانی" "Kurdiy Sorani"

!ifdef MUI_WELCOMEPAGE
  ${LangFileString} MUI_TEXT_WELCOME_INFO_TITLE "بەخێربێن بۆ یاریدەدەری دامەزراندنی $(^NameDA)"
  ${LangFileString} MUI_TEXT_WELCOME_INFO_TEXT "ئەم یاریدەدەرە لە دامەزراندنی $(^NameDA) ڕێنماییتان دەکات.$\r$\n$\r$\nپێشنیار دەکرێت هەموو بەرنامەکانی تر دابخەن پێش دەستپێکردنی دامەزراندن. ئەمە وا دەکات فایلە پێویستەکانی سیستەم بەبێ پێویستی بە دووبارە دەستپێکردنەوەی کۆمپیوتەرەکەتان نوێ بکرێنەوە.$\r$\n$\r$\n$_CLICK"
!endif

!ifdef MUI_UNWELCOMEPAGE
  ${LangFileString} MUI_UNTEXT_WELCOME_INFO_TITLE "بەخێربێن بۆ یاریدەدەری لابردنی دامەزراندنی $(^NameDA)"
  ${LangFileString} MUI_UNTEXT_WELCOME_INFO_TEXT "ئەم یاریدەدەرە لە لابردنی دامەزراندنی $(^NameDA) ڕێنماییتان دەکات.$\r$\n$\r$\nپێش دەستپێکردنی لابردنی دامەزراندن، دڵنیا ببنەوە لەوەی $(^NameDA) کار ناکات.$\r$\n$\r$\n$_CLICK"
!endif

!ifdef MUI_LICENSEPAGE
  ${LangFileString} MUI_TEXT_LICENSE_TITLE "ڕێککەوتننامەی مۆڵەت"
  ${LangFileString} MUI_TEXT_LICENSE_SUBTITLE "تکایە مەرجەکانی مۆڵەت بخوێننەوە پێش دامەزراندنی $(^NameDA)."
  ${LangFileString} MUI_INNERTEXT_LICENSE_BOTTOM "ئەگەر ڕازین بە مەرجەکانی ڕێککەوتننامەکە، بۆ بەردەوامبوون کرتە لە ڕازیم بکەن. دەبێت ڕێککەوتننامەکە قبوڵ بکەن بۆ دامەزراندنی $(^NameDA)."
  ${LangFileString} MUI_INNERTEXT_LICENSE_BOTTOM_CHECKBOX "ئەگەر ڕازین بە مەرجەکانی ڕێککەوتننامەکە، خانەی دیاریکردنی خوارەوە دیاری بکەن. دەبێت ڕێککەوتننامەکە قبوڵ بکەن بۆ دامەزراندنی $(^NameDA). $_CLICK"
  ${LangFileString} MUI_INNERTEXT_LICENSE_BOTTOM_RADIOBUTTONS "ئەگەر ڕازین بە مەرجەکانی ڕێککەوتننامەکە، یەکەم هەڵبژاردنی خوارەوە هەڵبژێرن. دەبێت ڕێککەوتننامەکە قبوڵ بکەن بۆ دامەزراندنی $(^NameDA). $_CLICK"
!endif

!ifdef MUI_UNLICENSEPAGE
  ${LangFileString} MUI_UNTEXT_LICENSE_TITLE "ڕێککەوتننامەی مۆڵەت"
  ${LangFileString} MUI_UNTEXT_LICENSE_SUBTITLE "تکایە مەرجەکانی مۆڵەت بخوێننەوە پێش لابردنی دامەزراندنی $(^NameDA)."
  ${LangFileString} MUI_UNINNERTEXT_LICENSE_BOTTOM "ئەگەر ڕازین بە مەرجەکانی ڕێککەوتننامەکە، بۆ بەردەوامبوون کرتە لە ڕازیم بکەن. دەبێت ڕێککەوتننامەکە قبوڵ بکەن بۆ لابردنی دامەزراندنی $(^NameDA)."
  ${LangFileString} MUI_UNINNERTEXT_LICENSE_BOTTOM_CHECKBOX "ئەگەر ڕازین بە مەرجەکانی ڕێککەوتننامەکە، خانەی دیاریکردنی خوارەوە دیاری بکەن. دەبێت ڕێککەوتننامەکە قبوڵ بکەن بۆ لابردنی دامەزراندنی $(^NameDA). $_CLICK"
  ${LangFileString} MUI_UNINNERTEXT_LICENSE_BOTTOM_RADIOBUTTONS "ئەگەر ڕازین بە مەرجەکانی ڕێککەوتننامەکە، یەکەم هەڵبژاردنی خوارەوە هەڵبژێرن. دەبێت ڕێککەوتننامەکە قبوڵ بکەن بۆ لابردنی دامەزراندنی $(^NameDA). $_CLICK"
!endif

!ifdef MUI_LICENSEPAGE | MUI_UNLICENSEPAGE
  ${LangFileString} MUI_INNERTEXT_LICENSE_TOP "کلیلی Page Down دابگرن بۆ بینینی ئەوەی ماوەتەوە لە ڕێککەوتننامەکە."
!endif

!ifdef MUI_COMPONENTSPAGE
  ${LangFileString} MUI_TEXT_COMPONENTS_TITLE "پێکهاتەکان هەڵبژێرن"
  ${LangFileString} MUI_TEXT_COMPONENTS_SUBTITLE "ئەو تایبەتمەندییانەی $(^NameDA) هەڵبژێرن کە دەتانەوێت دایانبمەزرێنن."
!endif

!ifdef MUI_UNCOMPONENTSPAGE
  ${LangFileString} MUI_UNTEXT_COMPONENTS_TITLE "پێکهاتەکان هەڵبژێرن"
  ${LangFileString} MUI_UNTEXT_COMPONENTS_SUBTITLE "ئەو تایبەتمەندییانەی $(^NameDA) هەڵبژێرن کە دەتانەوێت لایانببەن."
!endif

!ifdef MUI_COMPONENTSPAGE | MUI_UNCOMPONENTSPAGE
  ${LangFileString} MUI_INNERTEXT_COMPONENTS_DESCRIPTION_TITLE "وەسف"
  !ifndef NSIS_CONFIG_COMPONENTPAGE_ALTERNATIVE
    ${LangFileString} MUI_INNERTEXT_COMPONENTS_DESCRIPTION_INFO "نیشانەی مشکەکە ببەنە سەر پێکهاتەیەک بۆ بینینی وەسفەکەی."
  !else
    #FIXME:MUI_INNERTEXT_COMPONENTS_DESCRIPTION_INFO
  !endif
!endif

!ifdef MUI_DIRECTORYPAGE
  ${LangFileString} MUI_TEXT_DIRECTORY_TITLE "شوێنی دامەزراندن هەڵبژێرن"
  ${LangFileString} MUI_TEXT_DIRECTORY_SUBTITLE "ئەو بوخچەیە هەڵبژێرن کە دەتانەوێت $(^NameDA) تێیدا دابمەزرێنن."
!endif

!ifdef MUI_UNDIRECTORYPAGE
  ${LangFileString} MUI_UNTEXT_DIRECTORY_TITLE "شوێنی لابردنی دامەزراندن هەڵبژێرن"
  ${LangFileString} MUI_UNTEXT_DIRECTORY_SUBTITLE "ئەو بوخچەیە هەڵبژێرن کە دەتانەوێت $(^NameDA) لێی لابەرن."
!endif

!ifdef MUI_INSTFILESPAGE
  ${LangFileString} MUI_TEXT_INSTALLING_TITLE "دامەزراندن"
  ${LangFileString} MUI_TEXT_INSTALLING_SUBTITLE "تکایە چاوەڕێ بکەن تا $(^NameDA) دادەمەزرێت."
  ${LangFileString} MUI_TEXT_FINISH_TITLE "دامەزراندن تەواو بوو"
  ${LangFileString} MUI_TEXT_FINISH_SUBTITLE "دامەزراندنەکە بە سەرکەوتوویی تەواو بوو."
  ${LangFileString} MUI_TEXT_ABORT_TITLE "دامەزراندن هەڵوەشێنرایەوە"
  ${LangFileString} MUI_TEXT_ABORT_SUBTITLE "دامەزراندنەکە بە سەرکەوتوویی تەواو نەبوو."
!endif

!ifdef MUI_UNINSTFILESPAGE
  ${LangFileString} MUI_UNTEXT_UNINSTALLING_TITLE "لابردنی دامەزراندن"
  ${LangFileString} MUI_UNTEXT_UNINSTALLING_SUBTITLE "تکایە چاوەڕێ بکەن تا $(^NameDA) لادەبرێت."
  ${LangFileString} MUI_UNTEXT_FINISH_TITLE "لابردنی دامەزراندن تەواو بوو"
  ${LangFileString} MUI_UNTEXT_FINISH_SUBTITLE "لابردنی دامەزراندن بە سەرکەوتوویی تەواو بوو."
  ${LangFileString} MUI_UNTEXT_ABORT_TITLE "لابردنی دامەزراندن هەڵوەشێنرایەوە"
  ${LangFileString} MUI_UNTEXT_ABORT_SUBTITLE "لابردنی دامەزراندن بە سەرکەوتوویی تەواو نەبوو."
!endif

!ifdef MUI_FINISHPAGE
  ${LangFileString} MUI_TEXT_FINISH_INFO_TITLE "تەواوکردنی دامەزراندنی $(^NameDA)"
  ${LangFileString} MUI_TEXT_FINISH_INFO_TEXT "$(^NameDA) لەسەر کۆمپیوتەرەکەتان دامەزرا.$\r$\n$\r$\nبۆ داخستنی یاریدەدەرەکە کرتە لە تەواوکردن بکەن."
  ${LangFileString} MUI_TEXT_FINISH_INFO_REBOOT "پێویستە کۆمپیوتەرەکەتان دووبارە دەستپێبکاتەوە بۆ تەواوکردنی دامەزراندنی $(^NameDA). دەتانەوێت ئێستا دووبارە دەستپێبکاتەوە؟"
!endif

!ifdef MUI_UNFINISHPAGE
  ${LangFileString} MUI_UNTEXT_FINISH_INFO_TITLE "تەواوکردنی لابردنی دامەزراندنی $(^NameDA)"
  ${LangFileString} MUI_UNTEXT_FINISH_INFO_TEXT "$(^NameDA) لە کۆمپیوتەرەکەتان لابرا.$\r$\n$\r$\nبۆ داخستنی یاریدەدەرەکە کرتە لە تەواوکردن بکەن."
  ${LangFileString} MUI_UNTEXT_FINISH_INFO_REBOOT "پێویستە کۆمپیوتەرەکەتان دووبارە دەستپێبکاتەوە بۆ تەواوکردنی لابردنی دامەزراندنی $(^NameDA). دەتانەوێت ئێستا دووبارە دەستپێبکاتەوە؟"
!endif

!ifdef MUI_FINISHPAGE | MUI_UNFINISHPAGE
  ${LangFileString} MUI_TEXT_FINISH_REBOOTNOW "ئێستا دووبارە دەستپێبکەنەوە"
  ${LangFileString} MUI_TEXT_FINISH_REBOOTLATER "دەمەوێت دواتر بە دەستی دووبارە دەستپێبکەمەوە"
  ${LangFileString} MUI_TEXT_FINISH_RUN "&جێبەجێکردنی $(^NameDA)"
  ${LangFileString} MUI_TEXT_FINISH_SHOWREADME "&پیشاندانی Readme"
  ${LangFileString} MUI_BUTTONTEXT_FINISH "&تەواوکردن"  
!endif

!ifdef MUI_STARTMENUPAGE
  ${LangFileString} MUI_TEXT_STARTMENU_TITLE "بوخچەی لیستی دەستپێک هەڵبژێرن"
  ${LangFileString} MUI_TEXT_STARTMENU_SUBTITLE "بوخچەیەکی لیستی دەستپێک هەڵبژێرن بۆ کورتەبڕەکانی $(^NameDA)."
  ${LangFileString} MUI_INNERTEXT_STARTMENU_TOP "ئەو بوخچەیەی لیستی دەستپێک هەڵبژێرن کە دەتانەوێت کورتەبڕەکانی بەرنامەکەی تێدا دروست بکرێن. دەتوانن ناوێکیش بنووسن بۆ دروستکردنی بوخچەیەکی نوێ."
  ${LangFileString} MUI_INNERTEXT_STARTMENU_CHECKBOX "کورتەبڕ دروست مەکەن"
!endif

!ifdef MUI_UNCONFIRMPAGE
  ${LangFileString} MUI_UNTEXT_CONFIRM_TITLE "لابردنی دامەزراندنی $(^NameDA)"
  ${LangFileString} MUI_UNTEXT_CONFIRM_SUBTITLE "لابردنی $(^NameDA) لە کۆمپیوتەرەکەتان."
!endif

!ifdef MUI_ABORTWARNING
  ${LangFileString} MUI_TEXT_ABORTWARNING "دڵنیان لەوەی دەتانەوێت دامەزرێنەری $(^Name) دابخەن؟"
!endif

!ifdef MUI_UNABORTWARNING
  ${LangFileString} MUI_UNTEXT_ABORTWARNING "دڵنیان لەوەی دەتانەوێت لابەری دامەزراندنی $(^Name) دابخەن؟"
!endif

!ifdef MULTIUSER_INSTALLMODEPAGE
  ${LangFileString} MULTIUSER_TEXT_INSTALLMODE_TITLE "بەکارهێنەرەکان هەڵبژێرن"
  ${LangFileString} MULTIUSER_TEXT_INSTALLMODE_SUBTITLE "هەڵیبژێرن بۆ کام بەکارهێنەران دەتانەوێت $(^NameDA) دابمەزرێنن."
  ${LangFileString} MULTIUSER_INNERTEXT_INSTALLMODE_TOP "هەڵیبژێرن ئایا دەتانەوێت $(^NameDA) تەنها بۆ خۆتان دابمەزرێنن یان بۆ هەموو بەکارهێنەرانی ئەم کۆمپیوتەرە. $(^ClickNext)"
  ${LangFileString} MULTIUSER_INNERTEXT_INSTALLMODE_ALLUSERS "دامەزراندن بۆ هەموو بەکارهێنەرانی ئەم کۆمپیوتەرە"
  ${LangFileString} MULTIUSER_INNERTEXT_INSTALLMODE_CURRENTUSER "دامەزراندن تەنها بۆ خۆم"
!endif
