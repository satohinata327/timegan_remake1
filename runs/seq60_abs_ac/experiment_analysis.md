# seq60_abs_ac experiment analysis

## 概要

この実験では、`timegan_origin` のseq60 baselineに対して、Generator lossへ絶対リターン自己相関lossのみを追加した。

目的は、Mahalanobis距離の原因分析で強く出ていた

- `sp500_abs_autocorr_lag1`
- `sp500_abs_autocorr_lag5`
- `sp500_abs_autocorr_lag20`
- `dgs10_abs_autocorr_lag1`
- `dgs10_abs_autocorr_lag5`
- `dgs10_abs_autocorr_lag20`

を改善できるか確認することである。

追加したlossは、元スケールへ戻した `X` と `X_hat` に対して、

```text
Corr(|r_t|, |r_{t-k}|)
```

を `lag = 1, 5, 20` で計算し、実データwindowと生成windowの差をMSEで罰するものである。

## Mahalanobis評価結果

実データ参照windowのMahalanobis距離分布は以下である。

```text
min:  3.209714
mean: 4.391009
std:  0.732900
max:  7.107789
```

一方、`seq60_abs_ac` の生成データ10本はすべて参照分布の外側に位置した。

```text
distance min: 17.423457
distance max: 22.525545
reference_percentile: 100.00
empirical_upper_tail_probability: 0.000000
```

baselineの `timegan_origin/runs/seq60` と比較すると、

```text
origin seq60 mean distance:      11.310742
remake seq60_abs mean distance:  19.398606
```

となり、Mahalanobis距離は悪化した。

## 追加lossは効いているか

学習ログでは、joint training中の絶対リターン自己相関lossは以下のように低下した。

```text
step=1    abs_ac_loss=0.147239
step=300  abs_ac_loss=0.003810
step=1500 abs_ac_loss=0.002068
step=3000 abs_ac_loss=0.001637
```

したがって、追加したloss自体は最適化されている。

実際に、絶対リターン自己相関系の特徴量はbaselineより改善している。

```text
sp500_abs_autocorr_lag1
ref:    0.1937
origin: -0.0038
remake: 0.2033

sp500_abs_autocorr_lag5
ref:    0.2116
origin: 0.0010
remake: 0.1310

sp500_abs_autocorr_lag20
ref:    0.1267
origin: -0.0081
remake: 0.0699

dgs10_abs_autocorr_lag1
ref:    0.1475
origin: -0.0109
remake: 0.2045

dgs10_abs_autocorr_lag5
ref:    0.1449
origin: 0.0091
remake: 0.1250

dgs10_abs_autocorr_lag20
ref:    0.1026
origin: -0.0035
remake: 0.0718
```

この結果から、`abs_autocorr_loss` は狙った特徴量に対して一定の効果を持つと考えられる。

## 悪化した主な原因

Mahalanobis距離が悪化した主因は、2資産間の相関構造が大きく崩れたことである。

特に以下の特徴量が大きく悪化している。

```text
cross_corr
ref:    -0.0219
origin: -0.8229
remake:  0.9902

rolling_corr_std_60
ref:     0.2170
origin:  0.0252
remake:  0.0044

corr_down_sp500_q05
ref:     0.0752
origin: -0.3162
remake:  0.8595

corr_up_sp500_q95
ref:    -0.0458
origin: -0.3325
remake:  0.8213
```

`cross_corr` が約 `0.99` になっており、生成された `sp500` と `DGS10` がほぼ同じ方向に動く系列になっている。

また、`rolling_corr_std_60` が `0.0044` と非常に小さいため、この強い相関が時間的にもほぼ一定になっている。

つまり、生成データは2変量金融時系列として自然な関係を持つのではなく、2つの系列が過度に同調した形になってしまった。

## なぜこのような結果になったか

今回追加したlossは、各資産ごとに

```text
Corr(|r_t|, |r_{t-k}|)
```

を実データに近づけるものである。

しかし、このlossは以下を制約していない。

- `sp500` と `DGS10` の同時点相関
- rolling correlationの平均・標準偏差
- 下落局面・上昇局面での条件付き相関
- 2資産が互いに異なる系列として動くこと

そのため、モデルは各系列の絶対リターン自己相関を作ることには成功したが、2資産間の関係を自然に保つ方向には誘導されなかった。

むしろ、2系列を強く同調させることで、各系列の変動の大きさの持続性を作る簡単な解を選んだ可能性が高い。

このため、絶対リターン自己相関は改善した一方で、2資産間相関が極端になり、Mahalanobis距離全体としては悪化した。

## 結論

この実験から、以下がわかった。

```text
abs_autocorr_loss 単体は効く
ただし単体では危険
2資産間相関を同時に制約しないと、生成系列が過度に同調する
```

したがって、`abs_autocorr_loss` は捨てる必要はないが、単独で使うべきではない。

次の改良では、絶対リターン自己相関lossに加えて、少なくとも2資産間相関を制約するlossを同時に入れる必要がある。

## 次の方針

次の実装では、以下のようなlossを追加するのが自然である。

```text
L_G_total
= baseline TimeGAN loss
+ lambda_abs_ac * L_abs_ac
+ lambda_cross * L_cross
+ lambda_rollcorr * L_rolling_corr
```

優先候補は以下である。

```text
1. cross_corr loss
   Corr(sp500, DGS10) を実データwindowに近づける

2. rolling_corr_std_60 loss
   60日rolling correlationの変動幅を実データに近づける

3. conditional correlation loss
   sp500下位5%・上位5%局面での条件付き相関を近づける
```

特に今回の結果では、`cross_corr` が `0.99` まで上がっているため、次の叩き台ではまず `cross_corr loss` を追加することが重要である。
