# Data Collection

Wir konzentrieren uns bei den Weltraumdaten auf zwei Datensätze:

- SHARP
- NOAA

Für beide Datensätze liegen `fetch`-Skripte vor. Um die Zeitstempel der Daten zu harmonisieren, ist zusätzlich ein `sync` skript vorhanden.

Nach dem Synchronisieren haben die NOAA-Dateien mit dem `_harmonized`-Zusatz nur Zeitstempel, die zu denen aus dem SHARP-Datensatz passen.

## SHARP

- Datenpunkte alle 12 Minuten pro aktiver Region
- über 200 Spalten pro Datenpunkt, vermutlich nicht alle relevant
- ergänzt um die Spalten `timestamp` & `harp`

## NOAA

- mehrere Dateien
- Alle 1 bzw. 5 Minuten 1 Datenpunkt mit Messwerten wie
  - Magnetfeld
  - X-Ray
  - Electron Flux
  - ...
- Ergänzt jeweils um ein `_harmonized.csv`-File, wo die Werte der Daten passend zu den Zeitstempeln der `SHARP`-Daten sind
