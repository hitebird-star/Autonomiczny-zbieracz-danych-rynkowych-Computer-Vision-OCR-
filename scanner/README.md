# Scanner v10 — warstwa gry i capture

Nowy skaner działa obok `shop_scanner.py`. Plik legacy nie jest importowany ani
modyfikowany; v10 korzysta tylko z istniejącej kalibracji `scanner_config.json`.

## Polecenia

```powershell
cd "C:\Users\xxxxg\Desktop\Scraper Shop Glevia"

# Sklep musi być ręcznie otwarty. Sprawdza geometrię i zajęte sloty.
python -m scanner probe

# Powtarza hover na jednym zajętym slocie i podaje realny hit-rate.
python -m scanner hover-bench --attempts 10

# Przechwytuje ręcznie otwarty sklep do scans/<scan_id>/.
python -m scanner capture-open

# Capture + analiza Ollamą + eksport VERIFIED do ceny.csv.
python -m scanner capture-open --analyze

# Wykrywa i przechwytuje sklepy widoczne wokół postaci.
python -m scanner auto --max-shops 5

# Pełny obchód: capture kolejnego sklepu działa równolegle z analizą poprzedniego.
python -m scanner auto --walk --max-shops 20 --analyze

# Diagnostyka: zapis widoku, maski koloru, celow i kazdej proby klikniecia.
python -m scanner auto --walk --max-shops 3 --debug-live

# Dluzszy test bez edycji scanner_config.json: 4 pasy x 6 krokow = 27 ruchow.
python -m scanner auto --walk --max-shops 20 --lanes 4 --steps-per-lane 6 --debug-live
```

Parametry `--lanes`, `--steps-per-lane`, `--step-hold` i `--settle` nadpisuja
ustawienia `walk` tylko dla jednego uruchomienia. Log `session_start` zapisuje
faktyczny ksztalt trasy, a `session_end.end_reason` rozroznia
`max_shops_reached`, `route_exhausted`, `view_exhausted` i
`shop_close_blocked`.

## Diagnostyka autonomicznej petli

`--debug-live` tworzy katalog `dbg/auto/<data_godzina>/`:

```text
round_001_scene.png    # widok klienta uzyty przez detektor
round_001_mask.png     # biale = piksele uznane za kolor sklepu
round_001_overlay.png  # zielone = kandydaci, czerwone = odroczone, zolty = cel
events.jsonl           # klik, kursor, proba, grid_score, timeout i ruch WASD
```

Etykieta na overlay ma postac `numer/shop-id`; `/V` oznacza odwiedzony, a `/F`
terminalnie nieudany cel. Wykrywanie sklepow nie korzysta z OCR — ten podglad
pokazuje maske koloru, ktora faktycznie steruje wyborem punktu klikniecia.

Detektor K1 zachowuje propozycje z maski koloru, a lekki liniowy model
HOG+HSV jedynie ustawia ich kolejność. Cel podobny do postaci albo broni jest
oznaczany na czerwono i sprawdzany po celach bardziej podobnych do sklepu.
Model nie usuwa kandydatów i nie zmienia zbioru `max_results`, więc awaria lub
słaby wynik klasyfikatora nie może zmniejszyć liczby dostępnych sklepów. W
`events.jsonl` każdy kandydat ma `hybrid_score` i `likely_false`.
Zdarzenie `detection` zapisuje też `legacy_pick`: track, który wybrałby
detektor odległościowy spośród tego samego zbioru nieodwiedzonych celów.
`ranking_changed=true` oznacza czysty przypadek A/B, w którym hybryda wskazała
inny sklep. To pole jest wyłącznie diagnostyczne i nie zmienia działania bota.

Model można odtworzyć z regenerowalnego datasetu:

```powershell
python -m scanner.analysis.target_dataset
python -m scanner.detection.train_target_verifier
```

Brak plików `shop_target_svm.npz/json` automatycznie przywraca dokładnie
kolejność legacy: najbliższy cel jako pierwszy.

Każde polecenie daje trzy sekundy na przygotowanie gry, a potem aktywuje okno
`Glevia2` i przenosi kursor do bezpiecznego punktu wewnątrz klienta. Awaryjny
stop zapewnia PyAutoGUI: przesuń mysz w lewy górny róg.

Jeżeli Windows zablokuje automatyczny fokus, skaner czeka do 15 sekund na ręczne
kliknięcie `Glevia2`. Nie wykonuje zrzutów pulpitu ani IDE. `probe` zapisuje
`dbg/probe_grid.png` oraz `dbg/probe_overlay.png`; zielone ramki oznaczają zajęte
sloty, czerwone — puste. Gdy `open=False`, zajętość nie jest wyliczana.

Podczas skanowania nie klikaj PowerShella. Skaner wyłącza QuickEdit konsoli, a
przy utracie fokusu gry zatrzymuje się przed następnym slotem i wyświetla
`PAUZA: Glevia2 straciła fokus`. Wtedy kliknij ponownie grę — skan zostanie
wznowiony bez pomijania kolejnych dymków. `Esc` nie jest wysyłany do innego okna.

## Dane

Jeden sklep tworzy:

```text
scans/<scan_id>/
├── manifest.json
├── raw_events.jsonl
├── shop.png
└── tooltips/
    ├── slot_017_1.png
    └── slot_017_2.png
```

`manifest.json` jest podmieniany atomowo po każdym slocie. Capture nie czeka na
OCR ani model lokalny. Po stanie `CAPTURED` gra może przejść do kolejnego sklepu.

## Timingi v10

Capture ma własne szybkie wartości i nie dziedziczy starego
`timing.hover_delay=0.9`, potrzebnego wcześniej OCR-owi. Opcjonalne strojenie:

```json
{
  "capture_v10": {
    "hover_delay": 0.12,
    "move_duration": 0.05,
    "frames_per_slot": 1,
    "frame_interval": 0.035,
    "tooltip_timeout": 0.9,
    "tooltip_poll_interval": 0.05,
    "hover_attempts": 3,
    "hover_retry_delay": 0.08,
    "cursor_tolerance": 4,
    "tooltip_width": 620,
    "tooltip_height": 620
  }
}
```

Przed każdym slotem kursor przechodzi na pasek tytułu i zapisuje świeżą,
współpołożoną bazę obrazu. Nie wolno współdzielić jednej bazy między całym
sklepem: szeroki kadr zawiera animowaną scenę, więc po czasie różnica obrazu
przestaje izolować dymek i gwałtownie rośnie `tooltip_not_detected`.
Ruch jest wysyłany jako seria względnych zdarzeń Win32, które
klient Metin2 rejestruje pewniej niż pojedyncze ustawienie pozycji kursora.
Pozycja końcowa jest sprawdzana z tolerancją `cursor_tolerance`. Następnie skaner
czeka do `tooltip_timeout`, aż różnica obrazu potwierdzi pojawienie się panelu.
Przy braku dymka wykonuje maksymalnie `hover_attempts` pełnych cykli
`pasek tytułu → slot → względne drgnięcie`. Do analizy zapisuje ciasny wycinek
dymka, a nie całą scenę.

Po szybkim pierwszym przejściu wykonywany jest selektywny drugi przejazd tylko
po slotach zakończonych `tooltip_not_detected`. Nie zmienia on globalnych
`tooltip_timeout` ani `hover_attempts` i nie dotyka slotów już udanych.
Pierwszy przejazd używa najwyżej `first_pass_hover_attempts` prób
(domyślnie `2`), natomiast recovery zachowuje pełne `hover_attempts`
(domyślnie `3`). Dzięki temu trzecią, kosztowną próbę wykonują wyłącznie sloty,
które rzeczywiście trafiły do ogona recovery.
`raw_events.jsonl` zapisuje parę zdarzeń `recovery_pass`
(`phase=started/completed`) oraz `capture_pass=1|2` przy wyniku slotu.
Na początku drugiego przejazdu zapisuje również kompatybilne zdarzenie
`recovery_started` z liczbą `queued`, a wyniki drugiego przejazdu mają
`recovery_pass=true`. Ten minimalny format zasila offline harness
`scanner.analysis.recovery_audit`.
Odzyskana obserwacja ma evidence `tooltip_recovered_on_pass_2`.

Zajęte sloty są dodatkowo grupowane po środkowym wycinku ikony 20×20 px
(`icon_group`). Pierwszy slot grupy przechodzi pełną kontrolę markera ceny.
Kolejne sloty nadal są najeżdżane, ponieważ ta sama ikona może oznaczać inną
nazwę albo cenę. Dopiero praktycznie identyczny cały dymek korzysta z obrazu
reprezentanta i zapisuje zdarzenie `slot_deduplicated`. Każda zauważalna różnica
wraca do pełnej ścieżki OCR markera, więc matcher ikon nie jest źródłem prawdy
o ofercie.

`hover-bench` testuje jeden zajęty slot wielokrotnie, wypisuje `HIT/MISS`,
rzeczywistą pozycję kursora oraz geometrię okna. Udane klatki zapisuje do
`dbg/hover_bench/`. To podstawowy test po zmianie timingu lub sposobu ruchu.

Przy otwieraniu sklepu pierwsze kliknięcie zachowuje pełny `open_timeout`,
ponieważ postać może potrzebować czasu na podejście. Drugie kliknięcie ma
krótki, jednosekundowy timeout: po pierwszej próbie postać jest już przy celu,
a prawdziwe okno pojawia się szybko. Ogranicza to koszt płotów i postaci bez
usuwania retry potrzebnego części prawdziwych sklepów. Czas drugiej próby można
ustawić jako `timing.retry_open_timeout` (domyślnie `1.0`) w konfiguracji.

## Analiza lokalna

Flaga `--analyze` uruchamia `qwen3-vl:8b-instruct` przez Ollamę. Każdy dymek
otrzymuje dwa niezależnie przechwycone odczyty VLM oraz opcjonalne potwierdzenie
Windows OCR. Capture nie czeka na wynik pojedynczego dymka — worker analizuje
poprzedni sklep w tle.

Po zakończeniu obchodu polecenie czeka na opróżnienie kolejki analizy. Wyniki:

- `PROVISIONAL` — spójne, ale bez niezależnego potwierdzenia pola ryzyka;
- `VERIFIED` — potwierdzone; tylko one trafiają do `ceny.csv`;
- `REVIEW` — brak, konflikt albo niespójność danych.

### Eksport ofert

`ceny.csv` zawiera jeden wiersz dla tej samej oferty sprzedawcy (`item` +
cena za sztukę). `quantity` jest sumą dostępnych sztuk, a `stack_count` liczbą
osobnych slotów/stosów, które utworzyły ofertę. Slot bez licznika w prawym
dolnym rogu ma ilość `1`. Starsze wiersze otrzymują puste `stack_count`, bo
tej informacji nie da się uczciwie odtworzyć z samej sumy ilości.

Do diagnostyki bez żywej gry:

```powershell
python -m scanner.analysis --check
python -m scanner.analysis <scan_id>
python -m scanner.analysis --pending
```

## Granica modułów

`AnalysisWorker` przyjmuje dowolny obiekt realizujący:

```python
analyze(scan: ShopScan, repository: ScanRepository) -> ShopScan
```

Reader Ollamy i walidator są podpięte przez ten interfejs bez zależności od myszy,
ruchu i capture. CSV przyjmuje wyłącznie obserwacje o statusie `VERIFIED`.
