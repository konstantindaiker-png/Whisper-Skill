# Персональные заметки по диктовке/транскрибации (машина Константина)

Гайки и выученные через боль решения по этому инструменту. Раньше лежало в личной памяти Claude (`~/.claude/projects/.../memory/whisper-*.md`), вынесено сюда 21.06.2026, чтобы не грузить индекс памяти каждую сессию - читать, когда трогаешь именно диктовку/транскрибацию.

> Даты - точка во времени. Перед тем как утверждать как факт, сверяйся с текущим кодом/конфигом.

---

## Боевая конфигурация (источник истины)

Конфиг диктовки: `~/.config/whisper-skill/voice_dictation.json`. Актуальные значения (на 2026-06-12):
- хоткей - **правый Cmd (`<cmd_r>`)**, режим `ptt` (зажать-говорить-отпустить)
- `language: ru`, модель `large-v3-turbo`, `backend: faster`, `paste_mode: paste`, `mac_low_cpu_mode: true`

**Важно:** `DEFAULT_CONFIG` внутри `examples/voice_dictation.py` (`<ctrl>+<shift>+<space>`, `large-v3`, backend mlx) НЕ совпадает с реальной настройкой - Константин её перенастроил. Прежде чем называть хоткей/модель/режим - читать JSON-конфиг, а не дефолт из кода. Прецедент: 12.06 назвал хоткей по DEFAULT_CONFIG - он жал не ту клавишу.

Лог: `~/.config/whisper-skill/voice_dictation.log` (в нём персданные-надиктовки - наружу не слать).

---

## Диктовка в VS Code / Electron-webview

Основной сценарий - диктовать в chat-панель Claude Code внутри VS Code (Electron-webview).

- Симулированный Cmd+V через osascript в Electron-webview **молча игнорируется** (то же для Cursor chat, Slack, Discord, web-Chromium). Прямая печать символов работает.
- На macOS обходим `pynput.type()` полностью: шлём CGEvent с `keycode=0` + `CGEventKeyboardSetUnicodeString` напрямую через Quartz (`_macos_type_unicode` в `voice_dictation.py`); pynput оставлен как fallback. Этот путь игнорирует активную раскладку - иначе при русской системной раскладке pynput слал русские keycode'ы → в webview «ghbdtn» вместо «привет».
- НЕ переключать раскладку через TISSelectInputSource перед type() - race с pynput keycode mapping, делает только хуже.
- `dictate.sh` запускает `python3 -u` (без `-u` stdout буферизуется при редиректе в лог → диагностировать нечего). Маркеры в логе: `⌨ Typing N chars...` / `⌨ Typed N chars`.

Если «диктовка даёт билиберду» или «не работает»:
1. процесс жив: `ps aux | grep voice_dictation`
2. лог свежий: `stat -f "%m %N" ~/.config/whisper-skill/voice_dictation.log; date +%s`
3. `tail` лога - если текст в логе ПРАВИЛЬНЫЙ, а в чате мусор → проблема во вставке, не в распознавании
4. Quartz в venv: `python -c "import Quartz; print(Quartz.CGEventKeyboardSetUnicodeString)"`
5. оживление: `pkill -f voice_dictation; sleep 1; ~/.claude/skills/whisper-skill/dictate.sh > /tmp/whisper_dictation.log 2>&1 & disown`

---

## Висяк диктовки - корневая причина (pynput CGEventTap timeout)

Симптом: висит на `🎙 Recording...`, `on_release` не приходит, процесс жив, хоткей мёртв.

**Причина:** pynput слушает клавиатуру через `CGEventTapCreate`. macOS отключает tap навсегда при callback > ~1с (`kCGEventTapDisabledByTimeout`). Pynput это событие НЕ обрабатывает → `CFRunLoopRun` крутится впустую, `listener.running == True`, клавиатура мертва. Ловить disable-событие бесполезно (runloop занят зависшим callback'ом в момент timeout).

**Рабочее решение (2026-06-06):** не ждать события, а активно ОПРАШИВАТЬ tap. Watchdog (`_listener_watchdog`, ветка ptt) каждые 5с: `Quartz.CGEventTapIsEnabled(listener._tap)` → если False, `CGEventTapEnable(tap, True)` + форс-stop зависшей записи. Плюс страховка: `is_recording` дольше `max_recording_sec` (120с) → форс-stop. Конфиг-ключи: `watchdog_interval_sec`, `max_recording_sec`.

Если висяк повторится - искать в логе `⚠ CGEventTap was disabled — re-enabled by watchdog` или `⚠ recording stuck ... forcing stop`. Нет строк, а висяк есть → проверить, что `listener._tap` сохраняется (monkey-patch).

---

## Автозапуск при логине (macOS)

PTT-диктовка стартует сама при логине через launchd. Настроено 2026-06-13.

**Отдельный приватный репо:** `konstantindaiker-png/whisper-ptt-macos`. При переустановке Mac: `git clone … && ./install.sh`. В репо НЕТ бинаря, подписи и лога (персданные).

Артефакты на машине (раскладывает install.sh):
- `~/Applications/WhisperDictation.app/` - бандл. `Contents/MacOS/WhisperDictation` = нативный arm64-бинарь (исходник `Contents/Resources/launcher.c`), делает fork+exec `~/.claude/skills/whisper-skill/.venv/bin/python3 -m examples.voice_dictation`.
- `~/Library/LaunchAgents/com.konstantin.whisperdictation.plist` - агент (RunAtLoad + KeepAlive + ThrottleInterval=60), запускает бинарь напрямую.
- TCC выданы вручную на сам `WhisperDictation.app`: Accessibility + Input Monitoring + Microphone.
- Управление: `launchctl load/unload ~/Library/LaunchAgents/com.konstantin.whisperdictation.plist`.

Три грабли, почему «просто launchd на python» не работает:
1. **TCC на голый python слетает** - `.venv/bin/python3` симлинк на CLT python; разрешения инвалидируются при обновлении CLT (меняется cdhash). Лечение - стабильный bundle id у .app.
2. **shell-враппер крадёт TCC-identity** - процесс-образ становится `/bin/bash`, системному bash запрещён запрос микрофона → молчаливый denied → тишина → галлюцинация «Продолжение следует». Лечение - главный исполняемый = НАТИВНЫЙ C-лаунчер, не скрипт.
3. **Universal python берёт x86_64-срез** → arm64-библиотеки падают `incompatible architecture`. Лечение - лаунчер под arm64.

Доп: `open -W` в launchd упирается в single-instance LaunchServices - поэтому plist зовёт бинарь напрямую. ThrottleInterval=60 = страховка от краш-лупа.

---

## Транскрибация файлов - дефолты

- Модель по умолчанию **large-v3** уже скачена локально - не качать повторно, не предлагать small/medium «для скорости», не запускать wizard.py.
- Файлы Константин даёт сам (путь к локальному файлу). Из интернета (YouTube/TikTok) не качать, если явно не попросил.
- Большие файлы - всегда показывать прогресс/мониторинг (через `run_in_background` + проверки, либо вывод прогресса). Не запускать молча: иначе непонятно, идёт обработка или зависло.

---

## Русские длинные созвоны - рабочая связка

Для длинных RU-аудио (созвоны, паузы, шум) использовать **faster-whisper 1.2.1 + large-v3-turbo (CPU+int8) + Silero-VAD**, а НЕ голый mlx-whisper без VAD.

- Голый mlx без VAD циклится на тишине (`condition_on_previous_text=True`, нет VAD): «фокус фокус фокус», «ну ну ну», «Продолжение следует.». Silence-loop - свойство backend'а (mlx без VAD), не размера модели.
- Бенч на M4 (180с среза): large-v3 CPU = 296с (медленнее реалтайма), **turbo CPU = 68с (~4x быстрее)**, mlx-turbo GPU = 14с (~21x). Качество turbo не хуже large-v3 (точнее: «запрашиваю реквизиты» vs «прошиваю эквизиты»).
- Параметры: `WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cpu", compute_type="int8")`, `vad_filter=True`, `vad_parameters=dict(min_silence_duration_ms=500)`, `language="ru"`.
- Где применено: `board/transcribe_audio.py` (файловая транскрибация планёрок) переведён на turbo 2026-06-15. Диктовка (свой backend) - отдельно.
- Для коротких клипов/Reels - mlx turbo ок, циклы редки. Хочешь ещё быстрее на час аудио - mlx turbo на GPU M4, но нужен Silero-VAD пречанкинг руками.
- Модель `Systran/faster-whisper-large-v3` (CTranslate2) - отдельный файл от mlx-версии, в `~/.cache/huggingface/hub/`. Качается ~3 GB; при регулярном использовании поставить `HF_TOKEN`.
