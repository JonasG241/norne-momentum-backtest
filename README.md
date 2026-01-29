# norne-momentum-backtest
Backtesting of the momentum effect at OBX, with the purpose of finding a decision rule.


# strategier mtp portefølje:

A:
Overlay per aksje (for beslutningsstøtte)
Bruk score til å gi “inne/ute” (eller gradert) signal for én aksje.

Terskelregel: inne hvis score ≥ k, ute hvis score ≤ m

Potensielt, men mer komplisert: Gradert signal: høyere score = sterkere “hold/kjøp”-conviction, lav score = “reduser/selg”

Benchmark: buy-and-hold i samme aksje.

Altså bruk signal til å kjøpe inn og ut av enkelte aksjer, sammenlikn med buy and hold i samme aksje. Test, alle aksjer i ditt domene, hvilken strategi gir høyest gjennomsnittlig meravkastning.




B)
Likevektet portefølje (1/N)

Bygg en portefølje av aksjer som oppfyller signalet, likevekt mellom dem.

Alle over terskel: hold alle aksjer med score ≥ k

Top N: hold de N aksjene med høyest score

Bruk: viser om signalet fungerer “i snitt” på tvers av aksjer.




C)
Cap-vektet portefølje (indekslignende)

Samme seleksjon som over, men vekting etter markedsverdi/indeksvekter.

Cap-vekt blant aksjer med score ≥ k

(evt.) Indeksvekter blant valgte hvis dere har historiske indeksvekter

Bruk: mer direkte sammenliknbar med OBX/indeks.




# Score-varianter basert på 50/100/200 DMA

Dette notatet beskriver to score-metoder for å rangere trendstyrke basert på glidende snitt (50, 100 og 200 dager). Scorene kan brukes som input til både overlay per aksje og porteføljetester.

---

## 1) Alignment-score (0–3)

**Idé:** Tell hvor mange “bullish” relasjoner mellom glidende snitt som holder.

Definer indikatorer:
- `I_50_100 = 1` hvis `DMA50 > DMA100`, ellers `0`
- `I_100_200 = 1` hvis `DMA100 > DMA200`, ellers `0`
- `I_50_200 = 1` hvis `DMA50 > DMA200`, ellers `0`

**Alignment-score:**
- `Score_align = I_50_100 + I_100_200 + I_50_200`  (gir verdi `0..3`)

**Tolkning (intuisjon):**
- `3`: sterk bullish alignment (typisk `50 > 100 > 200`)
- `0`: sterk bearish alignment (typisk `50 < 100 < 200`)
- `1–2`: overgang/miks (delvis bullish, delvis bearish)

**Fordeler:**
- Enkel, transparent, skalerbar (kan utvides til flere snitt)
- Krever ikke “håndrangering” av alle permutasjoner

**Ulemper:**
- Skiller ikke alltid fint mellom alle mønstre som kan ha ulik praksisbetydning

---

## 2) Rekkefølge-score (permutasjonsscore)

**Idé:** Det finnes 6 mulige rekkefølger av (DMA50, DMA100, DMA200). Gi hver rekkefølge en score fra “mest bullish” til “mest bearish”.

**De 6 rekkefølgene:**
1. `50 > 100 > 200`
2. `50 > 200 > 100`
3. `100 > 50 > 200`
4. `100 > 200 > 50`
5. `200 > 50 > 100`
6. `200 > 100 > 50`

**Score-format:**
- `Score_rank ∈ {1..6}` (høyere = mer bullish) *eller* omvendt, men velg én standard og hold dere til den.

**Viktig:**
- Midtre rekkefølger (2–5) kan rangeres ulikt avhengig av prioritering (f.eks. om dere vektlegger at 200DMA skal ligge nederst/øverst).
- Derfor bør dere definere og dokumentere rangeringen *før* test (for å unngå “etterrasjonalisering”).

**Fordeler:**
- Mer granularitet enn alignment-score (skiller alle mønstre)
- Kan fange preferanser (f.eks. “200DMA nederst er viktig”)

**Ulemper:**
- Krever et subjektivt rangeringsvalg for “mellomtilfellene”
- Mindre skalerbar hvis dere senere legger til flere glidende snitt

---

## 3) Anbefalt praksis

- Bruk **alignment-score** som primærvariant (enkel, robust).
- Bruk **rekkefølge-score** som sekundærvariant for å sjekke robusthet og sensitivitet.
- Dokumentér:
  - eksakt definisjon av score,
  - hvilken retning som er “mer bullish”,
  - eventuelle tie-breaks (likhet mellom snitt) og hvordan dere håndterer dem.

---
