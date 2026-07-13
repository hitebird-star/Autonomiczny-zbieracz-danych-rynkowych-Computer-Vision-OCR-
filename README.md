# Autonomiczny-zbieracz-danych-rynkowych-Computer-Vision-OCR-

> Program w Pythonie, który samodzielnie porusza się po dynamicznym środowisku
> (żywy rynek w grze online na prywatnym serwerze), rozpoznaje obiekty na ekranie,
> odczytuje z nich dane cenowe i porządkuje je do dalszej analizy.
>
> **Nie jestem zawodowym programistą.** Ten projekt to mój pomysł, który doprowadziłem
> do działającego, złożonego systemu, **kierując pracą agentów AI** — projektując
> architekturę, podejmując decyzje inżynierskie, diagnozując błędy i iterując.
> To właśnie chcę tu pokazać: umiejętność orkiestracji AI i nadzoru nad własnym projektem.

🚧 **Projekt w aktywnym rozwoju.**

---

## O co chodzi

System działa jak automatyczny pipeline danych, tyle że źródłem nie jest baza czy plik,
tylko **żywy, ruchomy ekran**:

```
przechwycenie ekranu  →  rozpoznanie obiektów (Computer Vision)
      →  odczyt tekstu (OCR)  →  odszumianie i deduplikacja
      →  eksport do CSV  →  analiza
```

Środowiskiem testowym jest rynek w grze online (prywatny serwer, moje własne środowisko
do eksperymentów) — celowo trudne źródło danych: obraz jest zaszumiony, zmienny i w ruchu.
To było o wiele ciekawsze wyzwanie niż gotowy zbiór danych, bo cały „surowiec" trzeba
najpierw samemu zebrać z ekranu.

---

## Moja rola: pomysł + orkiestracja AI

Kod w dużej części powstał z pomocą agentów AI (Claude, Codex). Byłem szczery co do tego
od początku — i właśnie dlatego chcę precyzyjnie nazwać **mój faktyczny wkład**, bo to on
jest tu wartością:

- **Pomysł i prowadzenie projektu** — od pomysłu, przez kolejne wersje, po długoterminowy
  rozwój. To jeden, rozwijany od dawna system, a nie jednorazowy skrypt.
- **Projektowanie architektury** — podział na moduły (przechwytywanie / detekcja / analiza /
  zapis) i oddzielenie logiki „czystej" (dającej się testować bez gry) od warstwy wejścia/wyjścia.
- **Kierowanie wieloma agentami AI** z jasnym podziałem obowiązków, tak żeby sobie nie
  wchodziły w drogę i nie psuły nawzajem kodu (szczegóły niżej).
- **Diagnozowanie błędów z realnych danych** — nie „zgadywanie na czuja", tylko dochodzenie
  do przyczyny na podstawie zapisanych zrzutów i logów, a potem kierowanie AI prosto w źródło
  problemu.
- **Weryfikacja i jakość** — nie brałem outputu AI na wiarę: pilnowałem testów, wyłapywałem
  regresje i decydowałem, co wchodzi do projektu, a co nie.

Python znam na poziomie podstawowym — i tym cenniejsza jest umiejętność doprowadzenia
takiego systemu do działania **mimo to**: przez trafne decyzje, dobre kierowanie AI
i konsekwentną weryfikację.

---

## Jak zorganizowałem pracę z AI (to jest sedno)

To nie jest „wygenerowane jednym promptem". Nad projektem pracowało **dwóch agentów AI
naraz**, więc trzeba było tym realnie zarządzać:

- **Rozdzielone obszary odpowiedzialności („pasy").** Jeden agent odpowiadał za logikę
  analityczną (offline, testowalną bez gry), drugi za kod działający na żywo w grze
  (sterowanie, przechwytywanie obrazu, detekcja). Każdy pisał tylko w swoim obszarze.
- **Bez scalania gałęzi na ślepo.** Zmiany przenosiłem między agentami **plik po pliku,
  świadomie** — żeby jeden agent nie nadpisał ani nie zepsuł pracy drugiego. To była
  celowa dyscyplina, nie przypadek.
- **Testy jako siatka bezpieczeństwa.** Projekt ma **ponad 600 testów jednostkowych**.
  Traktowałem je jak kontrakt: jeśli zmiana psuła testy, wracała do poprawki, zanim weszła.
- **Iteracja na prawdziwych danych, nie na domysłach.** Zamiast pozwalać AI zgadywać,
  kazałem odtwarzać problem offline na realnych zrzutach ekranu i logach, aż było widać
  faktyczną przyczynę.

### Przykłady problemów, które rozwiązałem, kierując AI

- **„Bot kręcił się w kółko" (livelock).** System skanował w kółko to samo, zamiast iść
  dalej. Doszedłem, że przyczyną jest mylenie **tożsamości** obiektu z jego **pozycją**
  na ekranie (drgający obraz), a nie parametr dopasowania — i tam skierowałem poprawkę.
- **Błędna kalibracja mapy w regularnej siatce.** Rynek to krata niemal identycznych
  straganów, przez co dopasowywanie punktów dawało „ciche", błędne wyniki. Skierowałem
  rozwiązanie na inną metodę — analizę przesunięcia całej klatki obrazu (korelacja fazowa),
  odporną na tę powtarzalność.
- **OCR gubił współrzędną na słabym kontraście.** Zamiast w nieskończoność „stroić" OCR,
  **udowodniłem testem**, że strojenie tu nie pomoże — i dlatego kierunkiem stał się
  zapasowy odczyt lokalnym modelem wizyjnym.

---

## Co system potrafi

- Automatycznie rozpoznaje elementy interfejsu na ekranie (dopasowanie wzorców,
  sygnatury pikselowe).
- Odczytuje dane tekstowe z obrazu (OCR: Windows OCR, z lokalnym modelem wizyjnym
  jako odczytem zapasowym).
- Samodzielnie się porusza i podejmuje decyzje — z obsługą błędów i sytuacji brzegowych
  na zmiennych danych.
- Usuwa duplikaty i eksportuje uporządkowane dane do CSV do dalszej analizy.
- Buduje metryczną mapę rynku (osobny podsystem, wciąż rozwijany).

---

## Stack technologiczny

- **Język:** Python
- **Computer Vision:** OpenCV, NumPy, Pillow
- **OCR:** Windows OCR + lokalny model wizyjny (Ollama) jako odczyt zapasowy
- **Jakość:** ponad 600 testów jednostkowych, architektura modułowa (ponad 80 modułów)

---

## Status

Projekt w aktywnym rozwoju. Realnie działa i zbiera dane — w najdłuższym przebiegu
przetworzył setki obiektów w jednym uruchomieniu. Kolejny etap to podsystem budowy
dokładnej, metrycznej mapy rynku.

---

## Czego się nauczyłem

- Jak **rozłożyć duży pomysł na części** i doprowadzić go do działania, nie będąc
  zawodowym programistą.
- Jak **skutecznie kierować AI**: precyzyjnie stawiać zadania, weryfikować wyniki
  i nie brać niczego na wiarę.
- Jak **dochodzić do przyczyny problemu** na podstawie realnych danych, zamiast zgadywać.
- Jak pilnować, żeby projekt **nie rozpadał się przy zmianach** — testami i dyscypliną pracy.
