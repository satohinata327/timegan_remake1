# Mahalanobis feature cause analysis

この分析は、TimeGAN生成データのMahalanobis距離がどの特徴量によって押し上げられているかを調べるためのものです。

計算した指標:
- z-score: 各特徴量が実データwindow分布の平均から何標準偏差ずれているか。
- Mahalanobis寄与: `delta_i * (Sigma^-1 delta)_i`。全特徴量で合計するとMahalanobis距離の二乗になります。
- 1特徴量修正感度: その特徴量だけを実データ平均に戻したとき、Mahalanobis距離がどれだけ下がるか。

読み方:
- z-scoreは直感的なズレの大きさです。
- Mahalanobis寄与は共分散構造まで含めて距離を押し上げている度合いです。
- 修正感度は、その特徴量をlossで改善する価値の目安です。

## seq60_abs_ac_corr_tail5

- samples: 10
- distance mean: 22.708199
- distance std: 2.268932
- distance min/max: 19.418023 / 26.050914

優先度スコア上位:
- rolling_corr_std_60: priority_score=1.000000, group=cross_asset
- corr_down_sp500_q05: priority_score=0.806599, group=cross_asset
- dgs10_abs_autocorr_lag5: priority_score=0.739219, group=temporal_dependence
- sp500_q99: priority_score=0.710662, group=tail_quantile
- sp500_abs_autocorr_lag5: priority_score=0.614888, group=temporal_dependence
- dgs10_q01: priority_score=0.389533, group=tail_quantile
- corr_up_sp500_q95: priority_score=0.300572, group=cross_asset
- sp500_q01: priority_score=0.275700, group=tail_quantile
- dgs10_q99: priority_score=0.252659, group=tail_quantile
- sp500_q95: priority_score=0.111232, group=tail_quantile

Mahalanobis正寄与上位:
- rolling_corr_std_60: mean_positive_contribution_to_d2=90.207717, group=cross_asset
- sp500_q99: mean_positive_contribution_to_d2=83.544553, group=tail_quantile
- dgs10_abs_autocorr_lag5: mean_positive_contribution_to_d2=77.909184, group=temporal_dependence
- sp500_abs_autocorr_lag5: mean_positive_contribution_to_d2=75.023724, group=temporal_dependence
- corr_down_sp500_q05: mean_positive_contribution_to_d2=74.425249, group=cross_asset
- dgs10_q01: mean_positive_contribution_to_d2=53.538101, group=tail_quantile
- sp500_q01: mean_positive_contribution_to_d2=32.907836, group=tail_quantile
- dgs10_q99: mean_positive_contribution_to_d2=31.011529, group=tail_quantile
- corr_up_sp500_q95: mean_positive_contribution_to_d2=18.510198, group=cross_asset
- sp500_q95: mean_positive_contribution_to_d2=11.373482, group=tail_quantile

1特徴量修正による距離低下上位:
- rolling_corr_std_60: mean_distance_reduction=3.329997, group=cross_asset
- sp500_q99: mean_distance_reduction=2.525353, group=tail_quantile
- corr_down_sp500_q05: mean_distance_reduction=2.271147, group=cross_asset
- dgs10_abs_autocorr_lag5: mean_distance_reduction=2.263686, group=temporal_dependence
- sp500_abs_autocorr_lag5: mean_distance_reduction=1.657950, group=temporal_dependence
- dgs10_q01: mean_distance_reduction=0.743941, group=tail_quantile
- corr_up_sp500_q95: mean_distance_reduction=0.554468, group=cross_asset
- sp500_q95: mean_distance_reduction=0.371816, group=tail_quantile
- sp500_q01: mean_distance_reduction=0.240300, group=tail_quantile
- dgs10_q99: mean_distance_reduction=0.097250, group=tail_quantile

## 次のloss設計への示唆

- `priority_score` 上位の特徴量は、次の `timegan_remake1` でloss候補にする優先度が高いです。
- z-scoreだけでなくMahalanobis寄与と修正感度も高い特徴量は、単なる見かけのズレではなく距離そのものの主因になっている可能性が高いです。
- seq60とseq120の両方で上位に出る特徴量は、特定seq_lenへの過適合ではなく、より一般的な弱点として扱う価値があります。