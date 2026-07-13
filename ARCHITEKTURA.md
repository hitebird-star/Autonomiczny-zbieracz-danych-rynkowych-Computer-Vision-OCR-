# Architektura

Dokument pokazuje, **jak zaprojektowałem system** i **jak zorganizowałem pracę dwóch
agentów AI**, żeby sobie nie wchodziły w drogę. To jest sedno tego projektu — nie sam kod,
tylko sposób, w jaki nim pokierowałem.

---

## Zasada naczelna: rdzeń offline vs warstwa live

Cały system jest podzielony na dwie warstwy według jednego kryterium: **czy da się to
przetestować bez uruchamiania gry.**

- **Rdzeń offline (czysty)** — logika analityczna: OCR, parsowanie danych, deduplikacja,
  matematyka mapy. Nie dotyka ekranu ani klawiatury, więc jest w 100% testowalna
  (ponad 600 testów jednostkowych).
- **Warstwa live** — wszystko, co styka się z grą: przechwytywanie obrazu, sterowanie,
  detekcja na żywo, nawigacja.

Ten podział to nie kosmetyka — to on umożliwił jednoczesną pracę dwóch agentów AI bez
konfliktów i pozwolił łapać błędy testami, zanim trafiły „na żywo".

---

## Podział pracy między agentów AI („pasy")

```
        RDZEŃ OFFLINE (testowalny)      │      WARSTWA LIVE (gra)
     ───────────────────────────────   │   ───────────────────────────────
      analiza · OCR · dane · mapa       │   capture · detekcja · nawigacja
                                        │
              agent A                   │              agent B
        (logika, matematyka)            │       (sterowanie, ekran)
     ───────────────────────────────   │   ───────────────────────────────
                     └──────────  ja: nadzór  ──────────┘
        architektura · diagnoza · decyzje · weryfikacja · testy
```

**Zasady, które ustaliłem i których pilnowałem:**

1. **Każdy agent pisze tylko w swoim pasie.** Agent od analizy nie dotyka kodu gry i odwrotnie.
2. **Bez scalania gałęzi na ślepo.** Zmiany między agentami przenosiłem **plik po pliku,
   świadomie** — żeby jeden nie nadpisał pracy drugiego.
3. **Testy to kontrakt.** Zmiana, która psuła testy, nie wchodziła, dopóki nie była naprawiona.
4. **Diagnoza z realnych danych.** Problemy odtwarzaliśmy offline na zapisanych zrzutach
   ekranu i logach — zamiast zgadywać.

---

## Przepływ danych

```
  ┌─────────────┐   ┌──────────────┐   ┌───────────┐   ┌──────────────┐   ┌──────┐
  │ przechwycenie│──▶│  rozpoznanie │──▶│   odczyt  │──▶│ odszumianie  │──▶│ CSV  │
  │    ekranu    │   │  (Computer   │   │  tekstu   │   │      +       │   │  +   │
  │              │   │   Vision)    │   │   (OCR)   │   │ deduplikacja │   │analiza│
  └─────────────┘   └──────────────┘   └───────────┘   └──────────────┘   └──────┘
     capture/          detection/         analysis/        analysis/       storage/
```

---

## Mapa modułów

| Moduł | Warstwa | Odpowiedzialność |
|-------|---------|------------------|
| `scanner/capture/`    | live    | Przechwytywanie obrazu z okna gry, zrzuty tooltipów |
| `scanner/detection/`  | live    | Rozpoznawanie obiektów na ekranie (Computer Vision, dopasowanie wzorców) |
| `scanner/navigation/` | live    | Poruszanie się postaci, wyznaczanie trasy, granice obszaru |
| `scanner/analysis/`   | offline | OCR, odczyt współrzędnych i cen, deduplikacja, odczyt zapasowy modelem wizyjnym |
| `scanner/atlas/`      | offline | Budowa metrycznej mapy rynku: kalibracja, rejestracja klatek |
| `scanner/storage/`    | offline | Zapis danych: CSV, manifesty, repozytorium skanów |
| `scanner/models/`     | offline | Struktury danych (opis skanu sklepu itd.) |
| `scanner/pipeline.py` | spina   | Łączy warstwy w jeden przebieg: zobacz → rozpoznaj → odczytaj → zapisz |
| `scanner/app.py`      | spina   | Punkt wejścia, tryby uruchomienia, orkiestracja przebiegu |
| `win_ocr.py`          | offline | Cienki wrapper na wbudowany OCR Windows (PyWinRT) |

---

## Dlaczego akurat tak

- **Testowalność napędza projekt.** Im więcej logiki wepchnąłem do rdzenia offline, tym
  więcej dało się pokryć testami — a testy były jedyną realną obroną przed tym, żeby AI
  „naprawiając" jedno, nie zepsuło drugiego.
- **Odseparowane wejście/wyjście.** Gra jest nieprzewidywalna (zaszumiony, ruchomy obraz).
  Trzymając logikę z dala od ekranu, mogłem odtwarzać i naprawiać błędy na zapisanych
  danych, w spokoju, bez odpalania gry za każdym razem.
- **Podział na pasy = kontrola nad AI.** Dwa agenty na jednym kodzie bez granic to przepis
  na chaos. Wytyczone pasy sprawiły, że każda zmiana miała właściciela i dało się ją
  bezpiecznie przenieść.
