# STACK_AWARE_RECOVERY_SPEC — pomijanie redundantnych slotów w recovery

Autor specyfikacji: Claude (analiza offline). Implementacja: DeepSeek (pas live,
`scanner/pipeline.py` + `scanner/storage/scan_repository.py`). Ten plik NIE zmienia
kodu — opisuje *dokładnie* co i gdzie zmienić oraz kontrakt testów.

## 0. Dowód i cel

Pomiar 2 biegów (18:04 i 18:11 lok., 31 sklepów, `dark_share`):

| metryka | wartość |
|---|---|
| dymki slot-level | 83.4% (599/718) |
| porazek razem | 119 |
| porazek pokrytych stos-bratem | **90 (76%)** — cena już złapana z innego slotu tej samej ikony |
| naprawdę zgubione (unikat) | 29 (24%) |
| **efektywne pokrycie ofert** | **96.0%** |
| koszt recovery (mediana) | 16.8% czasu skanu |
| recovery yield | 7% (9/128) |

**76% slotów w kolejce recovery jest redundantnych** — ich `icon_group` ma już
brata ze statusem `CAPTURED`, więc cena/oferta jest już zarejestrowana. Recovery
grinduje (≈3 s/slot) o dane, które już mamy.

**Cel:** nie kolejkuj do recovery slotu, którego `icon_group` ma już złapanego
brata. Oczekiwany efekt: koszt recovery 16.8% → ~4%, dymki bez zmian (braki i tak
redundantne), grind tylko o realnie unikalne oferty.

## 1. Mechanizm — gotowe struktury w pipeline.py

Wszystko, czego trzeba, już istnieje w `_capture`/`capture_slot`:

- `group_by_slot[slot.slot]` → `icon_group` slotu, znany **przed** hoverem
  (z analizy ikon gridu). Każdy slot porażki ma `icon_group` (zweryfikowane:
  0 porażek bez grupy).
- `group_references: dict[icon_group, (frame0, paths, rep_slot)]` — wypełniane
  przy KAŻdym udanym capture (`group_references.setdefault(icon_group, …)`,
  pipeline.py:323). Po pass 1 zawiera każdą grupę, która ma złapanego członka.
  `rep_slot = group_references[g][2]` = slot reprezentanta.

Czyli „grupa ma złapanego brata" = `icon_group in group_references`.

## 2. Zmiana (pipeline.py, tuż przed pętlą recovery ~l.385)

**Haczyk czasowy:** reprezentant grupy bywa złapany PÓŹNIEJ w pass 1 niż jej brat,
który padł wcześniej. Dlatego filtr musi działać **po zakończeniu pass 1**
(na zebranym `failed_slots`), a NIE w momencie `failed_slots.append` (l.363).
Wtedy `group_references` jest już kompletne dla pass 1.

```python
# PO pass 1, PRZED recovery_started:
covered, recoverable = [], []
for slot in failed_slots:
    if group_by_slot[slot.slot] in group_references:
        covered.append(slot)
    else:
        recoverable.append(slot)

for slot in covered:
    rep = group_references[group_by_slot[slot.slot]][2]
    obs = scan.slots[slot.slot]          # już FAILED z pass 1
    obs.evidence = list(obs.evidence or []) + [f"stack_covered_by:{rep}"]
    self.repository.append_event(
        scan.scan_id, "recovery_skipped",
        slot=slot.slot, reason="stack_covered",
        icon_group=group_by_slot[slot.slot], covered_by=rep,
    )
self.repository.save_manifest(scan)

# recovery iteruje TYLKO recoverable:
if recoverable:
    self.repository.append_event(
        scan.scan_id, "recovery_started",
        queued=len(recoverable), stack_skipped=len(covered),
    )
    for index, slot in enumerate(recoverable, start=1):
        ...  # bez zmian (capture_pass=2, cap consecutive_failures)
```

**Opcjonalnie (drobny dodatkowy zysk):** w pętli recovery, na początku iteracji,
ponów check — jeśli `group_by_slot[slot.slot] in group_references` (rep złapany
w trakcie recovery), pomiń. Łapie przypadek, gdy recovery odzyska reprezentanta,
a kolejne sloty tej grupy stają się redundantne. Marginalne, ale tanie.

## 3. Status / rekord pominiętego slotu

Slot pominięty zostaje `FAILED` (jak był po pass 1), z dołożonym
`evidence = ["stack_covered_by:<rep_slot>"]`. NIE tworzymy nowego statusu —
warstwa konsensusu/CSV i tak agreguje po `icon_group`, więc FAILED-ale-pokryty
nie wnosi nic, a brat dostarcza cenę. To zgodne z D-002/konsensusem stosu.

## 4. Liczniki (żeby offline harness mierzył poprawnie)

- `recovery_started.queued` = `len(recoverable)` (po filtrze), nie surowe
  `len(failed_slots)` — inaczej `recovery_audit` policzy zawyżony yield/koszt.
- Dodaj `recovery_started.stack_skipped = len(covered)`.
- Nowe zdarzenie `recovery_skipped` (per slot) z `reason="stack_covered"`,
  `covered_by=<rep_slot>`. `recovery_audit` może je zsumować jako oszczędność.

## 5. Bezpieczeństwo (świadoma wymiana)

Ten sam `icon_group` ≈ ten sam item → zwykładniczo ta sama oferta. Ryzyko: sprzedawca
wystawił tę samą ikonę po dwóch różnych cenach jednostkowych w różnych slotach.
- Empirycznie genuine anomalia stosu = 0.08% (pomiar anomaly_cost).
- Recovery yield na tych slotach = 7% — i tak prawie nigdy byśmy ich nie odzyskali.
- Rozbieżność cen w grupie wyłapuje warstwa **konsensusu stosu** na ZŁAPANYCH
  członkach; pominięty slot i tak nie dodałby pewności.

Akceptowalne: 76% oszczędności kosztu recovery za ryzyko 0.08% na slotach o yieldzie 7%.

## 6. BLOKER pokrewny: v10.47 logowanie ramek jest zepsute

`pipeline.py` (~l.280-293) woła:
```python
self.repository.save_tooltip_image(scan.scan_id, f"{slot.slot}_baseline", 0, captured.baseline)
```
ale `save_tooltip_image` (scan_repository.py:92-96) waliduje `0 <= slot <= 99`
(int) i `frame >= 1`, oraz formatuje `f"slot_{slot:03d}"`. Z `slot="7_baseline"`
i `frame=0` to rzuca `TypeError`/`ValueError`, a wywołanie jest w `try/except
Exception: pass` → **ramki nigdy się nie zapisują** (stąd brak `*_baseline.png`
w skanach). Restart skanera nie pomoże.

**Fix (storage, pas DeepSeeka):** osobna metoda bez ograniczeń int/frame:
```python
def save_raw_frame(self, scan_id: str, name: str, image: Image.Image) -> str:
    return self._save_image(scan_id, f"frames/{name}.png", image)
```
i w pipeline:
```python
if captured.baseline is not None:
    self.repository.save_raw_frame(scan.scan_id, f"slot_{slot.slot:03d}_baseline", captured.baseline)
if captured.last_candidate is not None:
    self.repository.save_raw_frame(scan.scan_id, f"slot_{slot.slot:03d}_hover", captured.last_candidate)
```
**Usuń `try/except: pass`** albo loguj wyjątek — ciche połykanie ukryło tę awarię.

### Kontrakt ścieżki dla offline replay (uzgodniony z `detector_replay.py`)
- `scans/<id>/frames/slot_<NNN>_baseline.png`
- `scans/<id>/frames/slot_<NNN>_hover.png`
- **Dla rzetelnego A/B** loguj ramki także dla PRÓBKI sukcesów (np. co 5. udany
  slot), nie tylko porażek — inaczej harness nie zmierzy false-positive nowego
  detektora. (Wariant „ciemny ∩ stabilny w czasie" wymaga dodatkowo KILKU klatek
  hover, nie jednej — patrz `detector_replay.py`.)

## 7. Kontrakt testów (DeepSeek, na rdzeniu filtra)

Wydziel filtr do czystej funkcji `partition_recovery(failed_slot_ids, group_by_slot,
captured_groups) -> (recoverable, covered)` i przetestuj:

1. Slot, którego grupa ma brata CAPTURED → `covered`.
2. Slot z unikalną grupą (brak w captured_groups) → `recoverable`.
3. Rep złapany PÓŹNIEJ w pass 1 niż brat-porażka → brat trafia do `covered`
   (bo filtr po pass 1, nie w momencie failu).
4. Grupa bez żadnego sukcesu (cała padła) → wszystkie `recoverable`.
5. `icon_group is None` (gdyby się zdarzyło) → `recoverable` (nie zgadujemy).
