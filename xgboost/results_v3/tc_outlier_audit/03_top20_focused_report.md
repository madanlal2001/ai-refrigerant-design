# Tc Outlier Audit — Top 20 Focused Report (Frozen Protocol v3)

Ranked by XGB-D |error| on the frozen Tc test set (n=209). Descriptive diagnostics only — single frozen split, no causal claims.

'Elevated' = |error| under a variant exceeds that variant's own P75 of |error| over its full Tc test set. SYSTEMATIC = elevated in >=3/4 variants, MODEL-SENSITIVE = elevated in 1-2/4, NONE = elevated in 0/4 (can occur for a molecule pulled into this cohort by a different variant's own top list).

## Rank 1 — `FNF` (cluster 428)
- Actual Tc: 403.0 K
- XGB-A: pred=335.178  residual=+67.822  |error|=67.822 [ELEVATED]
- XGB-B: pred=356.999  residual=+46.001  |error|=46.001 [ELEVATED]
- XGB-C: pred=281.498  residual=+121.502  |error|=121.502 [ELEVATED]
- XGB-D: pred=276.703  residual=+126.297  |error|=126.297 [ELEVATED]
- Consistency: **SYSTEMATIC** (4/4 variants elevated)
- Domain of applicability: max Tanimoto to ANY train molecule = 0.2 (**LOW**), nearest = `FF`
- Tc-labeled-train nearest neighbor: `FF` (sim=0.2, actual Tc=144.19 K, gap=258.81 K)
- Cluster: 1 Tc-test molecule(s) of 1 total benchmark molecule(s) in cluster 428; cluster mean XGB-D |error| = 126.2973 K
- Structure: MW=53.01, rings=0 (aromatic=0), halogens=2, heteroatoms=3

## Rank 2 — `c1ccc(Nc2ccccc2)cc1` (cluster 143)
- Actual Tc: 931.15 K
- XGB-A: pred=804.007  residual=+127.143  |error|=127.143 [ELEVATED]
- XGB-B: pred=834.233  residual=+96.917  |error|=96.917 [ELEVATED]
- XGB-C: pred=807.428  residual=+123.722  |error|=123.722 [ELEVATED]
- XGB-D: pred=818.109  residual=+113.041  |error|=113.041 [ELEVATED]
- Consistency: **SYSTEMATIC** (4/4 variants elevated)
- Domain of applicability: max Tanimoto to ANY train molecule = 0.5294 (**MEDIUM**), nearest = `CNc1ccccc1`
- Tc-labeled-train nearest neighbor: `CNc1ccccc1` (sim=0.5294, actual Tc=701.52 K, gap=229.63 K)
- Cluster: 3 Tc-test molecule(s) of 3 total benchmark molecule(s) in cluster 143; cluster mean XGB-D |error| = 65.8388 K
- Structure: MW=169.23, rings=2 (aromatic=2), halogens=0, heteroatoms=1

## Rank 3 — `[HH]` (cluster 388)
- Actual Tc: 33.13 K
- XGB-A: pred=147.136  residual=-114.006  |error|=114.006 [ELEVATED]
- XGB-B: pred=178.057  residual=-144.927  |error|=144.927 [ELEVATED]
- XGB-C: pred=123.206  residual=-90.076  |error|=90.076 [ELEVATED]
- XGB-D: pred=137.965  residual=-104.835  |error|=104.835 [ELEVATED]
- Consistency: **SYSTEMATIC** (4/4 variants elevated)
- Domain of applicability: max Tanimoto to ANY train molecule = 0.1111 (**LOW**), nearest = `O=C(O)CCC(=O)O`
- Tc-labeled-train nearest neighbor: `O=C(O)CCC(=O)O` (sim=0.1111, actual Tc=851.0 K, gap=817.87 K)
- Cluster: 1 Tc-test molecule(s) of 1 total benchmark molecule(s) in cluster 388; cluster mean XGB-D |error| = 104.835 K
- Structure: MW=2.02, rings=0 (aromatic=0), halogens=0, heteroatoms=0

## Rank 4 — `CC(C)CCCC(C)CCCC(C)CCCCC(C)CCCC(C)CCCC(C)C` (cluster 10)
- Actual Tc: 796.0 K
- XGB-A: pred=899.151  residual=-103.151  |error|=103.151 [ELEVATED]
- XGB-B: pred=867.381  residual=-71.381  |error|=71.381 [ELEVATED]
- XGB-C: pred=895.38  residual=-99.38  |error|=99.38 [ELEVATED]
- XGB-D: pred=890.631  residual=-94.631  |error|=94.631 [ELEVATED]
- Consistency: **SYSTEMATIC** (4/4 variants elevated)
- Domain of applicability: max Tanimoto to ANY train molecule = 0.6842 (**HIGH**), nearest = `CC(C)CCCC(C)CCO`
- Tc-labeled-train nearest neighbor: `CC(C)CCCC(C)CCO` (sim=0.6842, actual Tc=632.18 K, gap=163.82 K) **[close neighbor, large label gap]**
- Cluster: 11 Tc-test molecule(s) of 11 total benchmark molecule(s) in cluster 10; cluster mean XGB-D |error| = 20.6651 K
- Structure: MW=422.83, rings=0 (aromatic=0), halogens=0, heteroatoms=0

## Rank 5 — `c1ccc(C(c2ccccc2)(c2ccccc2)c2ccccc2)cc1` (cluster 106)
- Actual Tc: 983.0 K
- XGB-A: pred=880.396  residual=+102.604  |error|=102.604 [ELEVATED]
- XGB-B: pred=902.818  residual=+80.182  |error|=80.182 [ELEVATED]
- XGB-C: pred=887.836  residual=+95.164  |error|=95.164 [ELEVATED]
- XGB-D: pred=893.761  residual=+89.239  |error|=89.239 [ELEVATED]
- Consistency: **SYSTEMATIC** (4/4 variants elevated)
- Domain of applicability: max Tanimoto to ANY train molecule = 0.45 (**MEDIUM**), nearest = `CCC(C)(c1ccccc1)C(C)(CC)c1ccccc1`
- Tc-labeled-train nearest neighbor: `CCC(C)(c1ccccc1)C(C)(CC)c1ccccc1` (sim=0.45, actual Tc=810.0 K, gap=173.0 K)
- Cluster: 9 Tc-test molecule(s) of 9 total benchmark molecule(s) in cluster 106; cluster mean XGB-D |error| = 26.4442 K
- Structure: MW=320.44, rings=4 (aromatic=4), halogens=0, heteroatoms=0

## Rank 6 — `O=S1(=O)CCCC1` (cluster 398)
- Actual Tc: 855.0 K
- XGB-A: pred=721.945  residual=+133.055  |error|=133.055 [ELEVATED]
- XGB-B: pred=754.006  residual=+100.994  |error|=100.994 [ELEVATED]
- XGB-C: pred=761.415  residual=+93.585  |error|=93.585 [ELEVATED]
- XGB-D: pred=778.996  residual=+76.004  |error|=76.004 [ELEVATED]
- Consistency: **SYSTEMATIC** (4/4 variants elevated)
- Domain of applicability: max Tanimoto to ANY train molecule = 0.1818 (**LOW**), nearest = `C1CCC1`
- Tc-labeled-train nearest neighbor: `C1CCC1` (sim=0.1818, actual Tc=459.94 K, gap=395.06 K)
- Cluster: 1 Tc-test molecule(s) of 1 total benchmark molecule(s) in cluster 398; cluster mean XGB-D |error| = 76.004 K
- Structure: MW=120.17, rings=1 (aromatic=0), halogens=0, heteroatoms=3

## Rank 7 — `O=[N+]([O-])Nc1ccccc1` (cluster 143)
- Actual Tc: 816.67 K
- XGB-A: pred=755.525  residual=+61.145  |error|=61.145 [ELEVATED]
- XGB-B: pred=780.376  residual=+36.294  |error|=36.294 [ELEVATED]
- XGB-C: pred=745.281  residual=+71.389  |error|=71.389 [ELEVATED]
- XGB-D: pred=757.091  residual=+59.579  |error|=59.579 [ELEVATED]
- Consistency: **SYSTEMATIC** (4/4 variants elevated)
- Domain of applicability: max Tanimoto to ANY train molecule = 0.5 (**MEDIUM**), nearest = `O=[N+]([O-])c1ccc(Nc2ccccc2)cc1`
- Tc-labeled-train nearest neighbor: `O=[N+]([O-])c1ccc(Nc2ccccc2)cc1` (sim=0.5, actual Tc=814.0 K, gap=2.67 K)
- Cluster: 3 Tc-test molecule(s) of 3 total benchmark molecule(s) in cluster 143; cluster mean XGB-D |error| = 65.8388 K
- Structure: MW=138.13, rings=1 (aromatic=1), halogens=0, heteroatoms=4

## Rank 8 — `Cc1cc(C(C)C)ccc1O` (cluster 102)
- Actual Tc: 785.78 K
- XGB-A: pred=727.565  residual=+58.215  |error|=58.215 [ELEVATED]
- XGB-B: pred=729.411  residual=+56.369  |error|=56.369 [ELEVATED]
- XGB-C: pred=730.809  residual=+54.971  |error|=54.971 [ELEVATED]
- XGB-D: pred=728.501  residual=+57.279  |error|=57.279 [ELEVATED]
- Consistency: **SYSTEMATIC** (4/4 variants elevated)
- Domain of applicability: max Tanimoto to ANY train molecule = 0.52 (**MEDIUM**), nearest = `Cc1ccc(O)c(C)c1`
- Tc-labeled-train nearest neighbor: `Cc1ccc(O)c(C)c1` (sim=0.52, actual Tc=707.7 K, gap=78.08 K)
- Cluster: 6 Tc-test molecule(s) of 6 total benchmark molecule(s) in cluster 102; cluster mean XGB-D |error| = 27.5264 K
- Structure: MW=150.22, rings=1 (aromatic=1), halogens=0, heteroatoms=1

## Rank 9 — `Cc1c(O)cccc1C(C)C` (cluster 102)
- Actual Tc: 785.78 K
- XGB-A: pred=722.708  residual=+63.072  |error|=63.072 [ELEVATED]
- XGB-B: pred=729.604  residual=+56.176  |error|=56.176 [ELEVATED]
- XGB-C: pred=722.113  residual=+63.667  |error|=63.667 [ELEVATED]
- XGB-D: pred=729.148  residual=+56.632  |error|=56.632 [ELEVATED]
- Consistency: **SYSTEMATIC** (4/4 variants elevated)
- Domain of applicability: max Tanimoto to ANY train molecule = 0.52 (**MEDIUM**), nearest = `Cc1cccc(O)c1C`
- Tc-labeled-train nearest neighbor: `Cc1cccc(O)c1C` (sim=0.52, actual Tc=722.93 K, gap=62.85 K)
- Cluster: 6 Tc-test molecule(s) of 6 total benchmark molecule(s) in cluster 102; cluster mean XGB-D |error| = 27.5264 K
- Structure: MW=150.22, rings=1 (aromatic=1), halogens=0, heteroatoms=1

## Rank 10 — `ICCCCCCCCCCI` (cluster 224)
- Actual Tc: 836.2 K
- XGB-A: pred=805.049  residual=+31.151  |error|=31.151 [ELEVATED]
- XGB-B: pred=786.724  residual=+49.476  |error|=49.476 [ELEVATED]
- XGB-C: pred=792.924  residual=+43.276  |error|=43.276 [ELEVATED]
- XGB-D: pred=784.67  residual=+51.53  |error|=51.53 [ELEVATED]
- Consistency: **SYSTEMATIC** (4/4 variants elevated)
- Domain of applicability: max Tanimoto to ANY train molecule = 0.875 (**HIGH**), nearest = `ICCCCI`
- Tc-labeled-train nearest neighbor: `ICCCCI` (sim=0.875, actual Tc=720.71 K, gap=115.49 K) **[close neighbor, large label gap]**
- Cluster: 3 Tc-test molecule(s) of 3 total benchmark molecule(s) in cluster 224; cluster mean XGB-D |error| = 23.7129 K
- Structure: MW=394.08, rings=0 (aromatic=0), halogens=2, heteroatoms=2

## Rank 11 — `COC(=O)CC(=O)OC` (cluster 9)
- Actual Tc: 646.0 K
- XGB-A: pred=734.584  residual=-88.584  |error|=88.584 [ELEVATED]
- XGB-B: pred=691.026  residual=-45.026  |error|=45.026 [ELEVATED]
- XGB-C: pred=685.476  residual=-39.476  |error|=39.476 [ELEVATED]
- XGB-D: pred=696.035  residual=-50.035  |error|=50.035 [ELEVATED]
- Consistency: **SYSTEMATIC** (4/4 variants elevated)
- Domain of applicability: max Tanimoto to ANY train molecule = 0.4545 (**MEDIUM**), nearest = `CCCCCCC(=O)OC`
- Tc-labeled-train nearest neighbor: `CCCCCCC(=O)OC` (sim=0.4545, actual Tc=628.0 K, gap=18.0 K)
- Cluster: 26 Tc-test molecule(s) of 26 total benchmark molecule(s) in cluster 9; cluster mean XGB-D |error| = 14.0885 K
- Structure: MW=132.12, rings=0 (aromatic=0), halogens=0, heteroatoms=4

## Rank 12 — `C[SiH](C)C` (cluster 458)
- Actual Tc: 432.0 K
- XGB-A: pred=503.626  residual=-71.626  |error|=71.626 [ELEVATED]
- XGB-B: pred=517.019  residual=-85.019  |error|=85.019 [ELEVATED]
- XGB-C: pred=482.577  residual=-50.577  |error|=50.577 [ELEVATED]
- XGB-D: pred=480.35  residual=-48.35  |error|=48.35 [ELEVATED]
- Consistency: **SYSTEMATIC** (4/4 variants elevated)
- Domain of applicability: max Tanimoto to ANY train molecule = 0.1667 (**LOW**), nearest = `CBr`
- Tc-labeled-train nearest neighbor: `CBr` (sim=0.1667, actual Tc=466.26 K, gap=34.26 K)
- Cluster: 1 Tc-test molecule(s) of 1 total benchmark molecule(s) in cluster 458; cluster mean XGB-D |error| = 48.3497 K
- Structure: MW=74.2, rings=0 (aromatic=0), halogens=0, heteroatoms=1

## Rank 13 — `CC(C)(C)c1cc(C(C)(C)C)cc(C(C)(C)C)c1` (cluster 106)
- Actual Tc: 700.5 K
- XGB-A: pred=732.968  residual=-32.468  |error|=32.468 [ELEVATED]
- XGB-B: pred=736.878  residual=-36.378  |error|=36.378 [ELEVATED]
- XGB-C: pred=725.762  residual=-25.262  |error|=25.262 [ELEVATED]
- XGB-D: pred=748.064  residual=-47.564  |error|=47.564 [ELEVATED]
- Consistency: **SYSTEMATIC** (4/4 variants elevated)
- Domain of applicability: max Tanimoto to ANY train molecule = 0.7333 (**HIGH**), nearest = `Cc1cc(C(C)(C)C)cc(C(C)(C)C)c1`
- Tc-labeled-train nearest neighbor: `Cc1cc(C(C)(C)C)cc(C(C)(C)C)c1` (sim=0.7333, actual Tc=695.5 K, gap=5.0 K)
- Cluster: 9 Tc-test molecule(s) of 9 total benchmark molecule(s) in cluster 106; cluster mean XGB-D |error| = 26.4442 K
- Structure: MW=246.44, rings=1 (aromatic=1), halogens=0, heteroatoms=0

## Rank 14 — `O=C1OC(=O)c2ccccc21` (cluster 408)
- Actual Tc: 810.0 K
- XGB-A: pred=761.047  residual=+48.953  |error|=48.953 [ELEVATED]
- XGB-B: pred=766.58  residual=+43.42  |error|=43.42 [ELEVATED]
- XGB-C: pred=753.49  residual=+56.51  |error|=56.51 [ELEVATED]
- XGB-D: pred=765.93  residual=+44.07  |error|=44.07 [ELEVATED]
- Consistency: **SYSTEMATIC** (4/4 variants elevated)
- Domain of applicability: max Tanimoto to ANY train molecule = 0.3684 (**LOW**), nearest = `c1ccc(-c2ccccc2-c2ccccc2)cc1`
- Tc-labeled-train nearest neighbor: `c1ccc(-c2ccccc2-c2ccccc2)cc1` (sim=0.3684, actual Tc=857.0 K, gap=47.0 K)
- Cluster: 1 Tc-test molecule(s) of 1 total benchmark molecule(s) in cluster 408; cluster mean XGB-D |error| = 44.0704 K
- Structure: MW=148.12, rings=2 (aromatic=1), halogens=0, heteroatoms=3

## Rank 15 — `C=C(CC(C)(C)c1ccccc1)c1ccccc1` (cluster 106)
- Actual Tc: 835.0 K
- XGB-A: pred=805.638  residual=+29.362  |error|=29.362 [ELEVATED]
- XGB-B: pred=798.637  residual=+36.363  |error|=36.363 [ELEVATED]
- XGB-C: pred=803.205  residual=+31.795  |error|=31.795 [ELEVATED]
- XGB-D: pred=795.89  residual=+39.11  |error|=39.11 [ELEVATED]
- Consistency: **SYSTEMATIC** (4/4 variants elevated)
- Domain of applicability: max Tanimoto to ANY train molecule = 0.4643 (**MEDIUM**), nearest = `CCC(C)(c1ccccc1)C(C)(CC)c1ccccc1`
- Tc-labeled-train nearest neighbor: `CCC(C)(c1ccccc1)C(C)(CC)c1ccccc1` (sim=0.4643, actual Tc=810.0 K, gap=25.0 K)
- Cluster: 9 Tc-test molecule(s) of 9 total benchmark molecule(s) in cluster 106; cluster mean XGB-D |error| = 26.4442 K
- Structure: MW=236.36, rings=2 (aromatic=2), halogens=0, heteroatoms=0

## Rank 16 — `CC(C)C(C)(CO)C(C)(C)C` (cluster 284)
- Actual Tc: 628.77 K
- XGB-A: pred=642.263  residual=-13.493  |error|=13.493
- XGB-B: pred=636.466  residual=-7.696  |error|=7.696
- XGB-C: pred=672.686  residual=-43.916  |error|=43.916 [ELEVATED]
- XGB-D: pred=667.843  residual=-39.073  |error|=39.073 [ELEVATED]
- Consistency: **MODEL-SENSITIVE** (2/4 variants elevated)
- Domain of applicability: max Tanimoto to ANY train molecule = 0.4762 (**MEDIUM**), nearest = `CC(C)C(C)(O)CO`
- Tc-labeled-train nearest neighbor: `CC(C)C(C)(O)CO` (sim=0.4762, actual Tc=636.35 K, gap=7.58 K)
- Cluster: 1 Tc-test molecule(s) of 1 total benchmark molecule(s) in cluster 284; cluster mean XGB-D |error| = 39.0728 K
- Structure: MW=158.28, rings=0 (aromatic=0), halogens=0, heteroatoms=1

## Rank 17 — `ClC1CCCCC1` (cluster 448)
- Actual Tc: 586.0 K
- XGB-A: pred=629.535  residual=-43.535  |error|=43.535 [ELEVATED]
- XGB-B: pred=637.333  residual=-51.333  |error|=51.333 [ELEVATED]
- XGB-C: pred=629.513  residual=-43.513  |error|=43.513 [ELEVATED]
- XGB-D: pred=624.228  residual=-38.228  |error|=38.228 [ELEVATED]
- Consistency: **SYSTEMATIC** (4/4 variants elevated)
- Domain of applicability: max Tanimoto to ANY train molecule = 0.375 (**LOW**), nearest = `CC1CCCCC1`
- Tc-labeled-train nearest neighbor: `CC1CCCCC1` (sim=0.375, actual Tc=572.185 K, gap=13.815 K)
- Cluster: 1 Tc-test molecule(s) of 1 total benchmark molecule(s) in cluster 448; cluster mean XGB-D |error| = 38.2282 K
- Structure: MW=118.61, rings=1 (aromatic=0), halogens=1, heteroatoms=1

## Rank 18 — `NCCOCCO` (cluster 15)
- Actual Tc: 721.0 K
- XGB-A: pred=709.332  residual=+11.668  |error|=11.668
- XGB-B: pred=696.404  residual=+24.596  |error|=24.596 [ELEVATED]
- XGB-C: pred=694.818  residual=+26.182  |error|=26.182 [ELEVATED]
- XGB-D: pred=684.154  residual=+36.846  |error|=36.846 [ELEVATED]
- Consistency: **SYSTEMATIC** (3/4 variants elevated)
- Domain of applicability: max Tanimoto to ANY train molecule = 0.4375 (**MEDIUM**), nearest = `NCCO`
- Tc-labeled-train nearest neighbor: `NCCO` (sim=0.4375, actual Tc=673.3 K, gap=47.7 K)
- Cluster: 18 Tc-test molecule(s) of 18 total benchmark molecule(s) in cluster 15; cluster mean XGB-D |error| = 17.7633 K
- Structure: MW=105.14, rings=0 (aromatic=0), halogens=0, heteroatoms=3

## Rank 19 — `c1cnoc1` (cluster 378)
- Actual Tc: 590.0 K
- XGB-A: pred=503.344  residual=+86.656  |error|=86.656 [ELEVATED]
- XGB-B: pred=537.263  residual=+52.737  |error|=52.737 [ELEVATED]
- XGB-C: pred=556.204  residual=+33.796  |error|=33.796 [ELEVATED]
- XGB-D: pred=553.414  residual=+36.586  |error|=36.586 [ELEVATED]
- Consistency: **SYSTEMATIC** (4/4 variants elevated)
- Domain of applicability: max Tanimoto to ANY train molecule = 0.2353 (**LOW**), nearest = `c1ccoc1`
- Tc-labeled-train nearest neighbor: `c1ccoc1` (sim=0.2353, actual Tc=489.96 K, gap=100.04 K)
- Cluster: 1 Tc-test molecule(s) of 1 total benchmark molecule(s) in cluster 378; cluster mean XGB-D |error| = 36.5856 K
- Structure: MW=69.06, rings=1 (aromatic=1), halogens=0, heteroatoms=2

## Rank 20 — `CCCOCCOCCO` (cluster 15)
- Actual Tc: 681.13 K
- XGB-A: pred=649.069  residual=+32.061  |error|=32.061 [ELEVATED]
- XGB-B: pred=652.135  residual=+28.995  |error|=28.995 [ELEVATED]
- XGB-C: pred=645.041  residual=+36.089  |error|=36.089 [ELEVATED]
- XGB-D: pred=645.756  residual=+35.374  |error|=35.374 [ELEVATED]
- Consistency: **SYSTEMATIC** (4/4 variants elevated)
- Domain of applicability: max Tanimoto to ANY train molecule = 0.4737 (**MEDIUM**), nearest = `CCCOCC`
- Tc-labeled-train nearest neighbor: `CCCOCC` (sim=0.4737, actual Tc=500.23 K, gap=180.9 K)
- Cluster: 18 Tc-test molecule(s) of 18 total benchmark molecule(s) in cluster 15; cluster mean XGB-D |error| = 17.7633 K
- Structure: MW=148.2, rings=0 (aromatic=0), halogens=0, heteroatoms=3
