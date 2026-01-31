# Žiadosť PERUN#2501 - Návrh projektu

## Názov projektu

Porovnávacia štúdia moderných architektúr difúznych modelov pre generovanie 3D mračien bodov s aplikáciou v bioinžinierstve

---

## Oblasť výskumu

Umelá inteligencia, bioinformatika, strojové učenie

---

## Abstrakt

Projekt sa zameriava na systematické porovnanie a vyhodnotenie moderných architektúr difúznych modelov pre generovanie 3D mračien bodov (point clouds). Difúzne modely predstavujú súčasný state-of-the-art prístup v generatívnom modelovaní, pričom ich aplikácia na 3D dáta je aktívnou oblasťou výskumu s významným potenciálom v biomedicínskom inžinierstve.

**Ciele projektu:**

1. Implementácia a tréning baseline difúzneho modelu PCDiff založeného na architektúre Point-Voxel CNN (PVCNN) s 1000 difúznymi krokmi
2. Implementácia a porovnanie pokročilých architektúr: Difix3D (vylepšená kondicionovaná difúzia), Acc3D (akcelerovaná difúzia) a Contrastive Energy Distillation (CED) pre jednokrokové generovanie
3. Vyhodnotenie trade-off medzi kvalitou generovania a výpočtovou náročnosťou jednotlivých prístupov
4. Validácia metód na reálnych biomedicínskych dátach

**Metodológia:**

Využijeme iteratívny denoising proces charakteristický pre difúzne modely, kde model postupne odstraňuje šum z náhodného vstupu až po finálny 3D tvar. Baseline model PCDiff využíva 1000 iterácií, čo je výpočtovo náročné. Preto budeme skúmať akcelerované metódy:

- **DDIM sampling** - deterministický sampling redukujúci počet krokov z 1000 na 50
- **Difix3D** - vylepšená architektúra s lepšou kondicionáciou na vstupné dáta
- **CED (Contrastive Energy Distillation)** - destilácia znalostí pre jednokrokové generovanie

**Aplikačná doména:**

Validáciu vykonáme na úlohe automatického dopĺňania defektov v 3D skenoch lebky - úloha relevantná pre plánovanie chirurgických zákrokov a návrh implantátov. Vstupom je mračno bodov reprezentujúce defektnú lebku (27 648 bodov), výstupom je mračno bodov reprezentujúce chýbajúcu časť (3 072 bodov). Následne aplikujeme voxelizačný model pre konverziu na 3D mesh.

**Očakávané výstupy:**

- Kvantitatívne porovnanie architektúr pomocou metrík: Dice Similarity Coefficient (DSC), boundary DSC (bDSC), Hausdorff Distance 95% (HD95)
- Analýza výpočtovej efektivity (čas inferencie, pamäťové nároky)
- Odporúčania pre výber architektúry v závislosti od požiadaviek aplikácie
- Vedecká publikácia sumarizujúca výsledky

---

## Zdôvodnenie využitia HPC prostriedkov

Tréning difúznych modelov pre 3D mračná bodov je extrémne výpočtovo náročný. Baseline model PCDiff vyžaduje 15 000 epoch trénovania s batch size 8 na jednej GPU, čo na bežnom hardvéri trvá približne 3 dni. Pre systematické porovnanie viacerých architektúr a hyperparametrových konfigurácií potrebujeme paralelizovať výpočty na viacerých GPU súčasne.

Multi-GPU tréning s 8 GPU umožňuje:

- Lineárne škálovanie batch size (8 → 64) pre stabilnejší tréning
- Redukciu času trénovania jedného modelu z 3 dní na menej ako 1 deň
- Súbežný tréning viacerých experimentálnych konfigurácií

Celkovo plánujeme natrénovať minimálne 4 rôzne architektúry, každú s 2-3 hyperparametrovými konfiguráciami, čo predstavuje 8-12 plných tréningových behov. Bez prístupu k HPC by tento výskum trval niekoľko mesiacov.

---

## Požadované zdroje

### CPU hodiny

**~43 000 CPU hodín** (2 týždne × 24h × 128 CPU)

### GPU hodiny

**~2 688 GPU hodín** (2 týždne × 24h × 8 GPU)

### Úložný priestor

**500 GB**

- Datasety: ~50 GB
- Checkpointy modelov: ~200 GB
- Výstupné dáta a logy: ~250 GB

---

## Požadované softvérové nástroje a knižnice

**Základné prostredie:**
- Python 3.11
- uv (moderný Python package manager)

**Deep Learning framework:**
- PyTorch 2.5+ s CUDA 12.4+ podporou
- torchvision
- torchaudio

**Špecializované knižnice pre 3D spracovanie:**
- PyTorch3D (Facebook Research) - operácie na 3D dátach
- torch-scatter - efektívne scatter operácie pre point clouds
- Open3D - vizualizácia a spracovanie 3D dát
- trimesh - práca s 3D meshmi

**Vedecké výpočty:**
- NumPy
- SciPy
- scikit-learn
- pandas

**Monitorovanie a logovanie:**
- Weights & Biases (wandb) - experiment tracking
- TensorBoard
- tqdm

**Ďalšie utility:**
- PyYAML - konfiguračné súbory
- h5py - efektívne ukladanie dát
- matplotlib - vizualizácie

**Poznámka:** Väčšinu Python knižníc si nainštalujeme pomocou uv do virtuálneho prostredia podľa potreby. Požadujeme len základnú inštaláciu CUDA toolkit a Python.

---

## Dáta pre výskum

Validáciu metód budeme vykonávať na reálnych anonymizovaných biomedicínskych dátach, ktoré nám na výskumné účely poskytli:

- **DimensionLab s.r.o.** - 3D skeny lebiek s defektmi
- **Biomedical Engineering s.r.o.** - referenčné dáta implantátov

Datasety obsahujú:
- **SkullBreak dataset**: 427 trénovacích vzoriek, 28 testovacích vzoriek
- Vstup: Defektná lebka ako mračno bodov (27 648 bodov)
- Výstup: Implantát ako mračno bodov (3 072 bodov)

---

## Členovia projektu

- **Hlavný riešiteľ:** [Meno PhD študenta z TUKE]
- **Spoluriešitelia:** [Doplniť podľa potreby]

---

## Časový plán

| Týždeň | Aktivita |
|--------|----------|
| 1 | Nastavenie prostredia, tréning baseline PCDiff modelu |
| 2 | Tréning alternatívnych architektúr (Difix3D, Acc3D, CED) |
| 3 | Evaluácia a porovnanie výsledkov |
| 4 | Analýza výsledkov, príprava publikácie |

---

## Referencie

1. Friedrich et al. (2023) - PCDiff: Diffusion Probabilistic Model for Point Cloud Completion
2. Luo & Hu (2021) - Diffusion Probabilistic Models for 3D Point Cloud Generation
3. Song et al. (2021) - Denoising Diffusion Implicit Models (DDIM)
