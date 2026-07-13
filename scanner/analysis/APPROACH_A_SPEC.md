# APPROACH A — spec ruchu dla DeepSeeka (po audycie biegu `--odometry`)

Zwięzła specyfika: jak zrobić, żeby bieg był approach A (a nie był). Powstała po audycie
sesji `dbg/auto/20260623_152750`. Czytaj z `HYBRID_SEAM.md` (wiring) — to jest „czego NIE robić".

## Dlaczego poprzedni bieg nie zadziałał (3 wady, twarde dane)

- Wężyk jechał poziom przez **'d'/'a'**, a one **OBRACAJĄ** (nie translują). Dowód: 8× d/a, a
  OCR został w blobie **X∈[401,411], Y∈[735,743] (~10×8)** — postać kręciła się w miejscu.
- Dead-reckon wziął magnitudę **3.2, ale zastosował SKALARNIE kartezjańsko** (d=+3.2x, a=−3.2x).
  Po 4×'a' DR=391.4, a OCR=407 → **błąd ~16 u** (wymyślił ruch poziomy, którego nie ma).
- `dense_stamp` niewpięty → rejestr **19/105 bez zmian** (zero densyfikacji).

## ✅ ZRÓB (approach A)

1. **Kamera FIXED** — ustaw raz na starcie, NIE obracaj podczas biegu.
2. **Ruch tylko 'W'** (przód = translacja); cofanie 'S'. **Żadnego d/a do jazdy.** Jedna linia.
3. **Dead-reckon WEKTOREM, nie skalarem:**
   ```python
   from scanner.analysis.odometry import Odometer, Calibration
   WALK = Calibration(vectors={"w": (-2.59, -1.87)})   # zastępuje hardcode 7.0/3.2-skalar
   odo  = Odometer(WALK)
   est  = odo.update("w", read_current_position())      # 'ocr' kotwica / 'dead_reckon' między
   ```
4. **Wepnij stamping i persystuj:**
   ```python
   from scanner.analysis.dense_stamp import decide_stamps
   for d in decide_stamps([s.fingerprint for s in detected], est,
                          envelope=_COORD_BOUNDS_FARM, max_dr_steps=3):
       registry.stamp(d.fingerprint, d.position, source=d.source)
   ```
5. **Guardy** przed/po kroku: `nav_guards.within_envelope(pos, _COORD_BOUNDS_FARM)`, `is_stuck`.

## ❌ NIE ROBNIJ

- **Żadnego d/a-jako-strafe** — to obroty (R²≈0 w kalibracji; blob 10×8 w biegu).
- **Żadnego skalarnego `units_per_step` po osi** (d=+x, a=−x, s=+y) — to stary zły model
  kartezjański, tylko z mniejszą liczbą. Rozjeżdża się z OCR o ~16 u.
- **Żadnego `SerpentineRoutePlanner`** dla tej gry — geometria d=prawo/a=lewo/s=dół jest
  nieważna dla tank-control. Do archiwum dla Glevia.
- **Nie oczekuj pokrycia 2D z A** — A to JEDNA linia (dowód pętli). 2D dopiero w B.

## 🔴 STUCK W TEKSTURZE — guard OBOWIĄZKOWY (powtarza się: bieg 154847, 161616)

Najczęstsza porażka biegów 'w': postać wchodzi w teksturę, stoi, a pętla pcha 'w' bez końca
(user musi Ctrl+C). **Dead-reckoning MASKUJE to zacięcie** — kluczowa pułapka:

- bieg 161616: kroki 1–4 OCR ~(456,746); od kroku 5 **OCR znika**, dead-reckon ślepo
  ekstrapoluje W-vector przez **12 kroków** do (425,724), choć postać tkwi w ścianie.
- `within_envelope` NIE łapie (425,724 jest W GRANICACH — tekstura wewnątrz farmy).
- `is_stuck` NIE łapie (dead-reckon fabrykuje ruch — pozycja „idzie" co krok).

### ⛔ PITFALL WIRINGU (bug 185827) — NIE bramkuj guardu za `pos is not None`

Realny bug: guard `fix_stale` był ZA `if pos is None: return` (pipeline.py:933) → gdy postać
w teksturze, OCR pada → `current_position=None` → metoda ucina się PRZED Guardem 3 →
`steps_since_fix` nie rośnie → `fix_stale` NIGDY nie strzela. **Guard od braku-pozycji wymagał
pozycji.** Dowód: biegi z 0 position_read (154847, 185827) = zacięcie; z reads (171528,181147)
= OK. **`pos is None` to NIE powód pominięcia — to SYGNAŁ.** Guard staleness MUSI biec
niezależnie od pos: licznik kroków-bez-fixu rośnie też (zwłaszcza!) gdy pos=None; resetuj go
tylko przy realnym fixie OCR. Najlepiej karm `fix_stale` przez `Odometer.Estimate.steps_since_fix`
(liczy się sam, też przed 1. kotwicą), nie ręcznym licznikiem schowanym za bramką.

**Sygnał = `Odometer.Estimate.steps_since_fix`** (kroki bez kotwicy OCR). Wepnij:
```python
from scanner.analysis import nav_guards as g
est = odo.update("w", read_current_position())
if g.fix_stale(est.steps_since_fix, max_steps=7):     # OCR milczy >7 krokow = stuck w teksturze
    stop_walk(reason=g.NavReason.LOST_FIX)            # i RECOVERY (akcja niżej)
if g.is_stuck(recent_OCR_positions):                  # KARM OCR, nie dead-reckon!
    stop_walk(reason=g.NavReason.STUCK)
```
⚠️ `is_stuck` karm **pozycjami OCR**, nie dead-reckonem (inaczej ślepy). `fix_stale` to główny
guard wewnętrznej przeszkody. `max_steps=7` > typowej dziury OCR (~4-5 na W-spacerze).

**RECOVERY (decyzja usera 23.06): cofnij 's' + obróć (a/d) + jedź dalej.** Po `fix_stale`:
1. **`s` ×N** (cofnij od przeszkody; N ~2-3, tunable live),
2. **`a` lub `d` ~90°** (obrót w nową stronę; ile kroków obrotu = tunable),
3. **wznów `w`** w nowym kierunku.

⚠️ **Skutek dla modelu:** obrót a/d **zmienił θ** → stary wektor W `(-2.59,-1.87)` jest już ZŁY.
Po recovery **odśwież kierunek z OCR** (θ z pierwszych 2 odczytów po wznowieniu, magnituda ≈3.2
zostaje) — to wymusza krok w stronę B. Odometer i tak re-kotwiczy na następnym OCR (dead-reckon
sprzed recovery był fikcją). `max_steps`/N/kąt obrotu — strojenie live (pas DeepSeek).

**Gotowy callable (Claude) — guard + akcja + odświeżenie kierunku:**
```python
from scanner.analysis import nav_guards as g
from scanner.analysis.odometry import heading_from_ocr

est = odo.update("w", read_current_position())
if g.fix_stale(est.steps_since_fix, max_steps=7):       # KIEDY (OCR milczy >7 krokow)
    for mv in g.recovery_plan(attempt):                 # CO: s xN -> a/d -> wznow w
        for _ in range(mv.count):
            movement.execute(mv.key, hold, settle)
    p0 = read_current_position()                        # 2 kotwice OCR po skrecie
    ...; p1 = read_current_position()                   # (kilka krokow 'w' miedzy)
    if p0 and p1:
        odo.set_vector("w", heading_from_ocr(p0, p1, magnitude=3.2))  # B: swiezy kierunek
```
`recovery_plan(attempt)` kręci naprzemiennie (1→'d', 2→'a'…) — gdy 1. próba nie odlepi, 2. w
drugą stronę. `heading_from_ocr(magnitude=3.2)` = kierunek świeży z OCR, długość stała.
`set_vector` rekalibruje dead-reckon bez gubienia pozycji/kotwicy.

## ⚠️ Kluczowy haczyk: wektor W zależy od kamery

`(-2.59,-1.87)` to kierunek 'w' DLA orientacji kamery z kalibracji. **Magnituda ≈3.2 jest
stała, KIERUNEK nie.** Najpewniej: na starcie biegu wyznacz wektor 'w' z **pierwszych 2
odczytów OCR** podczas marszu W (`θ=atan2(Δy,Δx)`, skala do 3.2) i tym karm `Calibration` —
wtedy A działa niezależnie od tego, jak ustawiona jest kamera. (To mini-krok w stronę B.)

## 🟥 DENSYFIKACJA NIE RUSZA — odometria rozłączona od stempla (audyt 23.06, plateau 39%)

**Objaw:** rejestr utknął **141/55 = 39%**. Czysty bieg po fixie wiringu (`20260623_191209`,
route_exhausted, **30 otwarć**, 27 position_read = 10 OCR + 17 dead-reckon) dołożył **0 nowych
pozycji** (0 rekordów `last_seen` 19:xx; `shops.jsonl` tylko przepisany).

**Root cause:** `_stamp_game_position` (pipeline.py:199 — JEDYNA ścieżka ustawiająca
`scan.game_position`) czyta coord OCR **świeżo w momencie otwarcia sklepu**:
```python
parsed = read_image(window_image)
if parsed is None:
    ...; return          # <-- OCR padł → brak stempla, koniec
```
Gdy ten OCR padnie → `return`, sklep zostaje bez pozycji. **Odometria `self._current_position`
(zna pozycję ZAWSZE — OCR-kotwica lub dead-reckon, 27 odczytów w biegu) NIE jest tu użyta.**
Dwa systemy rozłączone: chodzimy i wiemy gdzie jesteśmy, otwieramy sklep — a stempel próbuje
czytać OCR od zera i się poddaje. Stempel działa tylko gdy coord-OCR akurat trafi przy otwarciu
(~rzadko, por. Stage4), nie na każdym otwartym sklepie. To jest brakująca **środkowa warstwa**.

**FIX (twój pas, pipeline.py:~208) — fallback do odometrii gdy świeży OCR padnie:**
```python
parsed = read_image(window_image)
if parsed is None:
    est = self._current_position                 # odometria (OCR lub dead-reckon)
    if est is not None:
        scan.game_position = est
        self.repository.append_event(scan.scan_id, "game_position_stamped",
                                     x=est[0], y=est[1], source=self._position_source)
        return
    ...; return                                  # dopiero gdy i odometria pusta
```
Efekt: KAŻDY otwarty sklep dostaje pozycję (OCR jeśli jest, odometria jeśli nie) →
`positioned%` skacze z ~10% trafień-przy-otwarciu do ~100% otwartych. ⚠️ Tagguj source
(`ocr`/`dead_reckon`) w stemplu — pozycje dead-reckon są mniej pewne (filtr na potem).
`dense_stamp.decide_stamps` ma już bramkę `max_dr_steps` na to (odrzuca zbyt głęboki dead-reckon).

## ✅ Wiring staleness — POPRAWNY (potwierdzone 23.06)

`_steps_since_fix` rośnie przed bramką `pos is None` (climb zawsze), `fix_stale` sprawdzany
przed bramką, **reset tylko przy realnym OCR** (pipeline.py:948-950, `if position_source=="ocr"`).
Dead-reckon NIE resetuje → tekstura podbija licznik → recovery strzeli. Happy path przeżył
(route_exhausted). NIEPOTWIERDZONE: czy recovery realnie odpala W teksturze — bieg był za
zdrowy (OCR nie zamilkł >7). Trzeba biegu wchodzącego w teksturę (lub wymuś: zasłoń ROI coorda).

## ⚙️ Blocker infry: dysk

Bieg `20260623_190907` crash `OSError [Errno 28] No space left on device` (PIL nie zapisał
scene.png). `dbg/`=3.9 GB / 6.4 GB wolne. Czyść stare sesje dbg albo wyłącz zapis
scene/mask/overlay w auto — inaczej biegi padają losowo.

## 🛣️ DŁUGI BIEG — 3 wymagania (spec 23.06, pas DeepSeek = pipeline.py)

**Diagnoza czemu bieg się „przerywa":** kończy `route_exhausted` bo trasa = **JEDNA linia 'w'**.
Config nadpisał WSZYSTKIE klawisze trasy na 'w' (`walk.key_left/key_right/drop_key="w"`,
app.py:620-622) → 23× 'w' w jednym kierunku. Dane (bieg `194708`): trajektoria (444,735) →
(435,724) → … → (393,673), heading −144°, **skończyła na DOLNEJ krawędzi farmy** (y_min=672)
gdy wyszły kroki. To NIE bug — trasa z definicji to jedna prosta. **Wydłużenie (`--steps-per-lane`)
= dłuższa TA SAMA linia = wyjazd poza kopertę / w ścianę** (`lanes` nie tworzą zakrętów, bo
lane_key też = 'w'). Trzy rzeczy na długi, GĘSTY bieg:

### 1. FILAR B — trasa zawracająca (pokrycie 2D, dowolna długość)

Maszyneria JUŻ JEST: `_movement_recovery` (pipeline.py:899-913) robi obrót (`recovery_plan`) +
`heading_from_ocr` refresh W-vectora. Brakuje tylko **wyzwalania jej DELIBERATNIE na krawędzi
pasa** (nie tylko przy zacięciu).
- **Wzorzec:** jedź 'w' aż `within_envelope` wykryje **bliskość granicy** (margines ~1 W-step
  od y_min/y_max/x_min/x_max) → wykonaj **LANE-TURN** zamiast cichego STOP: obrót a/d (~180°
  wężyk) + `heading_from_ocr` (2 odczyty po skręcie) + wznów 'w'.
- **Dziś `within_envelope` na out-of-bounds robi cichy `return`** (linia 957-961) — NIE zatrzymuje
  biegu, tylko nie działa (bot jedzie 'w' dalej poza farmę). Zamień: bliskość/przekrok granicy
  → lane-turn (jeśli przekroczył margines: cofnij 's' do środka, potem obróć).
- **Lane-turn ≠ recovery:** recovery = reakcja na zacięcie; lane-turn = plan. Wspólny kod obrotu,
  ale inny reason w diagnostyce (`lane_turn` vs `recovery`) — inaczej audyt ich nie rozróżni.
- **Anty-pętla (KONIECZNE):** (a) histereza — po skręcie wymuś min. K kroków 'w' zanim znów
  dopuścisz turn (inaczej zakręca w kółko w rogu); (b) licz lane-turny, po N (≈ szerokość_farmy /
  W-step) zakończ `route_exhausted` ŚWIADOMIE = pokryłeś farmę; (c) nie zakręcaj 2× w tym samym
  rogu. Por. [[auto-walk-livelock]] — pętla obrotów to ten sam rodzaj livelocku.

### 2. POKRYCIE PER-MIEJSCE — nie porzucaj widoku z nieskanowanymi sklepami

**Problem (user: „duża część sklepów nieskanowana, idziemy dalej"):** `scan_current_view`
przerywa pętlę widoku na `stall_blocked` po **2 nieproduktywnych otwarciach z rzędu**
(duplikat/FAILED, pipeline.py:1079-1084) → reszta `next_unvisited` w tym widoku zostaje
niezeskanowana, bot robi krok 'w'. `stall_count` jest **GLOBALNY (per-view), nie per-shop** →
2 trudne sklepy z rzędu porzucają CAŁY widok, choć są łatwe nieodwiedzone. To źródło 46%
nieotwarć ([[mapping-rebuild-odometry]]).
- **Fix (rozróżnij livelock od niedokończonego pokrycia):**
  - licz stall **PER track_id** (ten sam sklep otwierany w kółko = livelock) zamiast globalnie; LUB
  - nie przerywaj dopóki `next_unvisited` zwraca **NOWY (inny)** track — przerwij dopiero gdy
    zostają same already-attempted; LUB
  - **retry FAILED-open R razy** zanim oznaczysz visited (część nieotwarć jest chwilowa).
- **Cel:** opróżnij widok (wszystkie wykryte sklepy spróbowane) ZANIM zrobisz krok trasy =
  gęstsze skany/miejsce.
- ⚠️ **Zachowaj escape z PRAWDZIWEGO livelocku** (jeden nieotwieralny sklep w kółko — to było
  po coś). Klucz rozróżnienia: livelock = powtarzanie JEDNEGO track_id, nie „2 trudne z rzędu".

### 3. PRÓBKOWANIE ITEMÓW — 1 na małe, 3 na duże (przyspieszenie)

`reps_for_size`/`select_representatives` JUŻ to robią (parametry `quorum`/`big_threshold`/
`big_quorum`) — warstwa Claude gotowa. Brakuje **przepchnięcia parametrów** przez
`enable_phase_b` (pipeline.py:105-109) → `select_representatives` (pipeline.py:402-403, dziś
przekazuje TYLKO `quorum`).
- **Twoja polityka** = `quorum=1, big_threshold≈5-8, big_quorum=3`: mały stos → 1 hover, duży → 3.
- **LICZBY** (pomiar `python -m scanner.analysis.representatives` na 924 manifestach, redundancja
  2,76×): polityka quorum=1 ≈ **60% mniej hoverów** (~2,5× szybsza faza hover) vs floor=3 (46%).
- 🔴 **TRADE-OFF (kontrakt konsensu §4):** quorum=1 = pojedynczy odczyt na mały stos → BRAK
  mniejszości demaskującej → cichy misread OCR (cena ×1000) **dziedziczy CAŁY stos**. Pomiar:
  ~**4119 slotów odsłoniętych**, 1506 grup bez konsensu. Schodzisz z floor=3 świadomie (user:
  „całość zliczalna przez LLM").
- **MITYGACJA (GOTOWA, pas Claude — `scanner/analysis/price_guard.py`, 26 testów):** zewnętrzny
  sanity-check zastępujący brakujący konsensus przy quorum=1. Po świeżym odczycie ceny itemu:
  ```python
  from scanner.analysis.price_guard import check_against_market, needs_rehover
  ref = [o.unit_price for o in offer_index.offers_for(item)]   # ceny tego itemu z INNYCH sklepow
  chk = check_against_market(price, ref, factor=10.0, min_samples=3)
  if needs_rehover(chk):           # HIGH/LOW/INVALID -> nie ufaj quorum=1, re-hover (wiecej repow)
      ...                          # chk.shift (np. 1000) = niemal pewny misread cyfr OCR
  ```
  Werdykt `HIGH`/`LOW` = sieć bezpieczeństwa (cena ≥10× lub ≤1/10 mediany rynku); `chk.shift`
  (±10/100/1000) = bonus „high confidence" gdy to round przesunięcie cyfr. `NO_REFERENCE` (<3
  cen) = item za świeży na ocenę — quorum=1 wtedy bez asekuracji (zostaw na floor=3 albo
  zaakceptuj). CLI audytu istniejących danych: `python -m scanner.analysis.price_guard`.
  Dopełnienie: **LLM-tally** robi sanity-check ceny vs reszta rynku, nie tylko liczy sztuki. Bez
  tego guardu quorum=1 przepuści ciche ×1000. Por. [[vlm-pipeline-value-test]] (yield vs poprawność
  — ale ×1000 to już poprawność), `offer_index.py` (skąd `ref`/mediana).

## Akceptacja (ten sam audyt powtórzę)

1. **dead-reckon trzyma się OCR** (rozbieżność mała, nie ~16 u),
2. **pozycje OCR rozciągają się w LINIĘ**, nie blob 10×8,
3. **`positioned%` w `shops.jsonl` rośnie** z 39% (po fallbacku odometrii — to klucz),
4. **kolejny bieg ma rekordy z bieżącym `last_seen`** (densyfikacja realnie ląduje, nie tylko przepisanie pliku).

Por. `HYBRID_SEAM.md`, `odometry.fit_translation` (skąd wektor), `dense_stamp`, `nav_guards`.
