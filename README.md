# timegan_remake1

`timegan_origin` をベースに、Generator lossへ **絶対リターン自己相関loss** を追加した叩き台実装です。

実験対象はまず `seq_len=60` のみです。出力形式は `each_mask/` と同じく、1260日分の `sp500,DGS10` リターンCSVです。

## 追加した方針

baselineのTimeGAN lossに、生成データ `X_hat` の絶対リターン自己相関が実データ `X` に近づくようなlossを追加します。

ラグ集合を

```text
K = {1, 5, 20}
```

とし、各資産 `j` について、

```text
rho_abs,j(k) = Corr(|r^j_t|, |r^j_{t-k}|)
rho_hat_abs,j(k) = Corr(|r_hat^j_t|, |r_hat^j_{t-k}|)
```

を計算します。

追加lossは、

```text
L_abs_ac =
mean_j mean_{k in K} (rho_hat_abs,j(k) - rho_abs,j(k))^2
```

です。

Generator側のlossは、

```text
L_G_total =
L_adv
+ 100 * sqrt(L_sup)
+ 100 * L_mom
+ lambda_abs_ac * L_abs_ac
```

になります。

`X_hat` と `X` は学習中はmin-max scaling後の値ですが、このlossではtorch上で学習データのmin/maxを使って元リターンスケールへ戻してから絶対リターン自己相関を計算します。

## ディレクトリ構成

```text
timegan_remake1/
  config/
    timegan_seq60_abs_ac.json
  data/
    train_sp500_us10y.csv
  mahalanobis_eval/
    scripts/run_mahalanobis_eval.py
  scripts/
    train_timegan.py
    generate_timegan.py
    evaluate_with_mahalanobis.py
    data_utils.py
    timegan_model.py
  runs/
    seq60_abs_ac/
      data/
      models/
      generated/
      evaluation/
      logs/
```

`models/` や `figures/` はトップ直下には作らず、実行後は `runs/seq60_abs_ac/` 以下に作られます。

## 環境構築

研究室GPUサーバーなどで、リポジトリの親ディレクトリにいる状態を想定します。

```bash
cd /path/to/DSS_code
python3 -m venv .venv
source .venv/bin/activate
pip install -r timegan_remake1/requirements.txt
```

CUDAが使えるか確認する場合:

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

`config/timegan_seq60_abs_ac.json` の `"device": "auto"` により、CUDAが使える場合はGPU、使えない場合はCPUを使います。

## 学習

```bash
python3 timegan_remake1/scripts/train_timegan.py \
  --config timegan_remake1/config/timegan_seq60_abs_ac.json
```

学習済みモデルは以下に保存されます。

```text
timegan_remake1/runs/seq60_abs_ac/models/timegan_abs_autocorr.pt
```

学習ログは以下です。

```text
timegan_remake1/runs/seq60_abs_ac/logs/train_log.txt
```

## 生成

```bash
python3 timegan_remake1/scripts/generate_timegan.py \
  --config timegan_remake1/config/timegan_seq60_abs_ac.json \
  --checkpoint timegan_remake1/runs/seq60_abs_ac/models/timegan_abs_autocorr.pt \
  --scaler timegan_remake1/runs/seq60_abs_ac/data/scaler.json
```

生成データは以下に保存されます。

```text
timegan_remake1/runs/seq60_abs_ac/generated/
```

## Mahalanobis評価

`timegan_remake1` 内にコピーしたMahalanobis評価コードを使うため、`mahalanobis_remake1/` には依存しません。

```bash
python3 timegan_remake1/scripts/evaluate_with_mahalanobis.py \
  --generated-dir timegan_remake1/runs/seq60_abs_ac/generated \
  --work-mask-dir timegan_remake1/runs/seq60_abs_ac/evaluation/mahalanobis_input \
  --output-dir timegan_remake1/runs/seq60_abs_ac/evaluation/mahalanobis_results \
  --mahalanobis-script timegan_remake1/mahalanobis_eval/scripts/run_mahalanobis_eval.py \
  --train-csv timegan_remake1/data/train_sp500_us10y.csv
```

評価結果は以下に出ます。

```text
timegan_remake1/runs/seq60_abs_ac/evaluation/mahalanobis_results/
```

## 最小手順

```bash
cd /path/to/DSS_code
source .venv/bin/activate

python3 timegan_remake1/scripts/train_timegan.py \
  --config timegan_remake1/config/timegan_seq60_abs_ac.json

python3 timegan_remake1/scripts/generate_timegan.py \
  --config timegan_remake1/config/timegan_seq60_abs_ac.json \
  --checkpoint timegan_remake1/runs/seq60_abs_ac/models/timegan_abs_autocorr.pt \
  --scaler timegan_remake1/runs/seq60_abs_ac/data/scaler.json

python3 timegan_remake1/scripts/evaluate_with_mahalanobis.py \
  --generated-dir timegan_remake1/runs/seq60_abs_ac/generated \
  --work-mask-dir timegan_remake1/runs/seq60_abs_ac/evaluation/mahalanobis_input \
  --output-dir timegan_remake1/runs/seq60_abs_ac/evaluation/mahalanobis_results \
  --mahalanobis-script timegan_remake1/mahalanobis_eval/scripts/run_mahalanobis_eval.py \
  --train-csv timegan_remake1/data/train_sp500_us10y.csv
```

## 実験メモ

この版では、あえて `cross_corr` や `rolling_corr_std_60` のlossはまだ入れていません。

まずは `abs_autocorr_loss` だけを追加し、baselineの `timegan_origin` と比較して、特に以下の特徴量が改善するかを確認します。

- `sp500_abs_autocorr_lag1`
- `sp500_abs_autocorr_lag5`
- `sp500_abs_autocorr_lag20`
- `dgs10_abs_autocorr_lag1`
- `dgs10_abs_autocorr_lag5`
- `dgs10_abs_autocorr_lag20`

特に原因分析で強く出ていた `sp500_abs_autocorr_lag5` が改善するかが重要です。
