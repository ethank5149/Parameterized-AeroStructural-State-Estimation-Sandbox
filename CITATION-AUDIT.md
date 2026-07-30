# Citation audit — PASSES / PASSES-HGV

Every entry in `passes-references.bib` and `passes-hgv-references.bib` was
checked against publisher records, DOI registries, or NTRS. Audit date:
2026-07-30.

**Result: 8 entries carried confirmed errors, 1 could not be verified at all.**

---

## Confirmed errors, now corrected

### 1. `mehra1970` — two different papers conflated (severity: high)

The entry paired the **title** of Mehra's 1972 paper with the **volume, issue,
and page range** of his 1970 paper. Both are real; they are different works.

| Field | Was | Correct |
|---|---|---|
| title | "Approaches to adaptive filtering" | belongs to the 1972 paper |
| year / vol / no / pp | 1970 / 15 / 2 / 175–184 | belongs to "On the identification of variances and adaptive Kalman filtering" |

Both are now present as `mehra1970` (IEEE TAC 15(2):175–184, doi
`10.1109/TAC.1970.1099422`) and `mehra1972` (IEEE TAC 17(5):693–698, doi
`10.1109/TAC.1972.1100100`). This matters because the entry is load-bearing for
the IAE method — the variance-identification lineage is the 1970 paper.

### 2. `hsu2018` — wrong journal, wrong pages, wrong author initials (severity: high)

Cited as *Journal of Computational Physics* **365**:257–270 (2018). No such
article exists. The real paper is:

> C. P. Hsu, C. F. Hung, J. Y. Liao, "A Chebyshev Spectral Method with Null
> Space Approach for Boundary-Value Problems of Euler–Bernoulli Beam,"
> *Shock and Vibration*, vol. 2018, Article ID 2487697.
> doi `10.1155/2018/2487697`

Author initials in the old entry (`Hsu, J.`, `Hung, T.`, `Liao, C.`) were also
wrong. This entry carries the entire Null Space Approach argument in both
papers, so a fabricated venue here would have been the single most damaging
citation error in the manuscripts.

### 3. `moyer1967` → `moyer1968` — wrong year, incomplete title (severity: medium)

NASA CR-1061 is dated **June 1968**, not 1967, and is **Part II** of a
four-part Aerotherm series. The full title is now given. Key renamed to
`moyer1968`; **update `\cite{moyer1967}` → `\cite{moyer1968}` in the .tex.**

### 4. `lees1956` — wrong page range (severity: low)

Was 259–274. The article runs 259–**269**. *Jet Propulsion* 26(4),
doi `10.2514/8.6977`.

### 5. `olver2013` — wrong page range, and the two bib files disagreed (severity: medium)

`passes-references.bib` had 462–489 (correct). `passes-hgv-references.bib` had
439–459 (wrong). Same paper, two different page ranges across your own
bibliography — exactly the kind of inconsistency a copy-editor catches.
Correct: *SIAM Review* 55(3):462–489, doi `10.1137/120865458`.

### 6. `dec2006` — wrong title (severity: medium)

Given as "An approximate solution for multi-dimensional charring material
thermal response." No Dec & Braun 2006 paper by that title exists. The 2006
paper is "An approximate ablative thermal protection system sizing tool for
entry system design," AIAA Paper 2006-780, doi `10.2514/6.2006-780`. The
`volume = {2006}, pages = {1-14}` fields were also malformed for an AIAA paper.

### 7. `huntington2008` — wrong title, missing volume/pages (severity: medium)

Given as "Optimal nonlinear trajectory generation" with no volume or pages.
The Huntington & Rao 2008 JGCD paper is "Optimal reconfiguration of spacecraft
formations using the Gauss pseudospectral method," JGCD 31(3):689–698,
doi `10.2514/1.31083`.

### 8. `abramson1966` — wrong contributor role, truncated title (severity: low)

Abramson is the **editor** of the NASA SP-106 monograph, not sole author, and
the full title includes ", with Applications to Space Vehicle Technology."

---

## Could not be verified — replaced

### `pei2011` — no such publication found

> Pei, J., "Slosh dynamics in aerospace applications," *Progress in Aerospace
> Sciences*, 2011.

No volume or pages were given, and the entry does not appear in Scopus,
Crossref, ScienceDirect, or Google Scholar under that title, author, journal,
or year. **Treat this as fabricated until you can produce the PDF.** If you
have a source for it, restore it; otherwise it should stay out.

Replaced with a verified Pei paper on the same subject:
Pei, J., "Analytical Investigation of Propellant Slosh Stability Boundary on a
Space Vehicle," *J. Spacecraft and Rockets* 58(5):1514–1521 (2021),
doi `10.2514/1.A35024`.

---

## Verified correct (no change beyond formatting/DOI)

`crank1984`, `shyy1996`, `trefethen2000`, `voller1981`, `barshalom2001`,
`zarchan2012`, `driscoll2016`, `chen1999`, `dodge2000`, `mohamed1999`,
`hide2003`, `elnagar1995`, `benson2006`, `fay1958`, `mindlin1951`,
`chen2020`, `mao2016`, `groves2013`, `vallado2013`.

Note on `zarchan2012`: the 6th edition (2012, Progress in Astronautics and
Aeronautics vol. 239, ISBN 978-1-60086-894-8) is correct as cited. A **7th
edition** now exists — consider citing it instead if you have access.

Note on `voller1981`: indexes disagree on the closing page (556 vs. 566).
Retained as 545–556, the range that appears in the ScienceDirect record.
Low stakes, but confirm against the PDF in `reference/` if you cite it.

---

## Newly added entries (all verified before insertion)

Added to support gaps identified in the manuscripts:

| Key | Reason | Record |
|---|---|---|
| `mehra1972` | disambiguates the conflated entry | IEEE TAC 17(5):693–698 |
| `sutton1971` | Sutton–Graves is named in the HGV intro but was never cited | NASA TR R-376 (Nov 1971) |
| `tauber1991` | Tauber–Sutton radiative heating is in the README but absent from the paper | JSR 28(1):40–42 |
| `reissner1945` | Mindlin–Reissner theory was cited to Mindlin only | JAM 12:A69–A77 |
| `townsend2015` | the bivariate ultraspherical/Kronecker construction the HGV paper actually uses | JCP 299:106–123 |
| `szmuk2020` | current 6-DoF SCvx reference; `mao2016` alone is the convergence-theory paper | JGCD 43(8):1399–1413 |
| `chen1999` | added to the HGV bib for the CMA/FIAT comparison | JSR 36(3):475–483 |
| `moyer1968` | added to the HGV bib to resolve the Stefan-vs-CMA inconsistency | NASA CR-1061 |
| `knox2026passes` / `knox2026hgv` | the two papers are a series and must cite each other | placeholder — **insert arXiv IDs once posted** |

---

## Outstanding action

- `\cite{moyer1967}` in `passes-updated.tex` must become `\cite{moyer1968}`.
- Insert arXiv identifiers into `knox2026passes` / `knox2026hgv` after posting.
- Confirm `voller1981` closing page against the PDF in `reference/`.
- Produce a source for `pei2011` or leave it out permanently.
