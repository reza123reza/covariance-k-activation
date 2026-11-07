# Covariance-based k activation function — results summary

**English** · [فارسی](#فارسی)

**Result:** the idea was not confirmed. After about 1,900 runs on three datasets, updating k with a covariance rule was no better than learning k normally with gradients. The one solid finding: moving k in the *negative* covariance direction is stable, while the positive direction makes the model blow up.

## Idea

- Formula: `f(x) = k + ReLU(x)`; every neuron has its own k, starting at zero.
- k is not learned by gradient. After each weight update it is updated from the running covariance between x and grad_x.
- Two update rules were tested: (a) `k = s·Cov` and (b) `k += lr·Cov`.

## Implementation

- Model: MLP with two 2048-unit layers, Adam, batch size 4096, on a T4 GPU in Colab.
- k is a buffer (it does not take part in backprop); grad_x is captured with a hook.
- Covariance is computed from running sums (Σx, Σg, Σxg); grad_x is multiplied by the batch size so it is not too small.
- AMP (float16 compute) was used for speed; its loss scale is removed from the gradient.

## Early MNIST experiments

- With a small coefficient (100), k stayed near zero (about 0.001) and accuracy matched ReLU (about 98.4%).
- Coefficients 1e4 and 1e5 blew up (loss = nan). Cause: a feedback loop — larger k, larger gradient, larger k again.
- A normalized version (`Cov / RMS(g)`) was stable but gave no improvement.

## Full bias-free MNIST experiment

- 6 formulas (`k+relu`, `relu(x−k)`, `k+leaky`, `prelu_k`, `k+softplus`, `k+silu`) × 7 k-methods × 3 seeds.
- No combination beat ReLU; the larger k became, the lower the accuracy.
- Same without bias: MNIST does not need a bias at all (98.27% with bias vs 98.28% without), so k had nothing to do.

## Full sweep: positive/negative k direction × three datasets (1,782 runs)

Each covariance method was tested with a positive (`+Cov`) and a negative (`−Cov`) coefficient, no bias, 3 seeds, 10 epochs.

| Dataset | ReLU with bias | Best with k | Verdict |
|---|---|---|---|
| MNIST | 98.21% | 98.29% | no effect |
| Fashion-MNIST | 88.80% | 88.72% | no effect |
| CIFAR-10 | 52.85% | 54.20% (`relu(x−k)` with `cov` +3000) | looked +1.3% better, only 3 seeds |

- Blow-ups: on MNIST 41 with `+Cov` vs 7 with `−Cov`; on Fashion-MNIST 49 vs 7.
- On CIFAR-10, for `k+relu` and `k+leaky` the negative direction was better in all 5 methods; for `relu(x−k)` the positive direction (because k enters the formula with a minus sign). So in both cases the shift is −s·Cov.
- Softplus with the negative direction improved 1–6% on all three datasets, but still stayed behind ReLU.

## Confirmation experiment on CIFAR-10 (10 models × 10 seeds × 30 epochs)

The covariance rule performed the same as learning k by gradient; the earlier +1.3% gain did not replicate.

| Model | Last-epoch accuracy (mean ± std) |
|---|---|
| SiLU + bias | 57.01% ± 0.63 |
| SiLU with covariance (`cov −100`) | 57.00% ± 0.54 |
| `k+relu` with covariance (`cov_norm −1`) | 55.13% ± 0.73 |
| `relu(x−k)` with gradient (`learned`) | 55.03% ± 0.45 |
| `relu(x−k)` with covariance (`cov_tanh +1e4`) | 55.01% ± 0.69 |
| `k+relu` with gradient (`learned`) | 55.00% ± 0.51 |
| `k+relu` with covariance (`cov_ema −1`) | 54.93% ± 0.71 |
| `relu(x−k)` with covariance (`cov +3000`) | 54.87% ± 1.23 |
| ReLU without bias | 54.54% ± 0.81 |
| ReLU + bias (reference) | 54.39% ± 0.61 |

- Covariance vs `learned` with the same formula: 0 to 0.13%, within random noise.
- The reference overfit more than the others (from 55.39% at its best epoch to 54.39%); much of the others' edge over it comes from that.
- With 9 comparisons, p of roughly 0.02–0.05 is not reliable.

## Analysis

1. In `k+relu` and `relu(x−k)`, the next layer's bias can take the place of k, so k adds no new capability; its only possible effect comes from how training behaves. The confirmation experiment showed covariance is no better than gradients there either.
2. The direction of k's movement matters: negative covariance was always more stable; the positive direction blew up many times more often.
3. MNIST and Fashion-MNIST hit their accuracy ceiling and show no difference.
4. The apparent +1.3% on CIFAR-10 came from few seeds and picking the best of about 33 settings.

## Final conclusion

- The idea was not confirmed: the covariance rule is no better than normal gradient-learned k.
- Side finding: if k is updated by covariance, only the negative direction is stable.
- Side finding: SiLU was about 2.5% better than ReLU on CIFAR-10, but that has nothing to do with k.

## Code

Scripts used: `shifted_relu_mnist.py`, `k_activation_sweep.py`, `k_activation_sweep_nobias.py`, `k_sweep_full.py`, `k_confirm_cifar10.py` (not included in this repository).

## License

MIT — see [LICENSE](LICENSE).

---

<div dir="rtl">

## فارسی

# تابع فعال‌سازی با k مبتنی بر کوواریانس — خلاصه‌ی نتایج

**نتیجه:** ایده تأیید نشد. بعد از حدود ۱۹۰۰ اجرا روی سه دیتاست، آپدیت k با کوواریانس بهتر از یادگیری عادی k با گرادیان نبود. تنها یافته‌ی قطعی: حرکت k در جهت منفیِ کوواریانس پایدار است و جهت مثبت مدل را منفجر می‌کند.

## ایده

- فرمول: `f(x) = k + ReLU(x)`؛ هر نورون k جداگانه دارد و k از صفر شروع می‌شود.
- k با گرادیان یاد گرفته نمی‌شود. بعد از هر آپدیت وزن‌ها، از روی کوواریانس تجمعی بین x و grad_x آپدیت می‌شود.
- دو روش آپدیت تست شد: الف `k = s·Cov` و ب `k += lr·Cov`.

## پیاده‌سازی

- مدل: MLP با دو لایه‌ی ۲۰۴۸تایی، Adam، batch ۴۰۹۶، روی GPU T4 در کولب.
- k یک buffer است (در گرادیان شرکت نمی‌کند)؛ grad_x با یک hook گرفته می‌شود.
- کوواریانس با جمع‌های تجمعی (Σx، Σg، Σxg) حساب می‌شود؛ grad_x در اندازه‌ی batch ضرب می‌شود تا خیلی کوچک نباشد.
- برای سرعت از AMP (محاسبه با float16) استفاده شد؛ scale آن از گرادیان برداشته می‌شود.

## آزمایش‌های اولیه روی MNIST

- با ضریب کوچک (۱۰۰)، k تقریباً صفر ماند (حدود ۰.۰۰۱) و دقت همان ReLU بود (حدود ۹۸.۴٪).
- ضریب‌های 1e4 و 1e5 منفجر شدند (loss = nan). علت: حلقه‌ی بازخورد؛ k بزرگ‌تر، گرادیان بزرگ‌تر، دوباره k بزرگ‌تر.
- نسخه‌ی نرمال‌شده (`Cov / RMS(g)`) پایدار بود ولی بهبودی نداد.

## آزمایش جامع و بدون bias روی MNIST

- ۶ فرمول (`k+relu`، `relu(x−k)`، `k+leaky`، `prelu_k`، `k+softplus`، `k+silu`) × ۷ روش k × ۳ seed.
- هیچ ترکیبی بهتر از ReLU نشد؛ هرچه k بزرگ‌تر شد، دقت کمتر شد.
- بدون bias هم همین بود: MNIST اصلاً به bias نیاز ندارد (۹۸.۲۷٪ با bias در برابر ۹۸.۲۸٪ بدون bias)، پس k هم کاری برای انجام دادن نداشت.

## آزمایش کامل: جهت مثبت و منفی k × سه دیتاست (۱۷۸۲ اجرا)

هر روش کوواریانس با ضریب مثبت (`+Cov`) و منفی (`−Cov`)، بدون bias، با ۳ seed و ۱۰ epoch تست شد.

| دیتاست | ReLU با bias | بهترین نتیجه با k | نتیجه |
|---|---|---|---|
| MNIST | ۹۸.۲۱٪ | ۹۸.۲۹٪ | بی‌اثر |
| Fashion-MNIST | ۸۸.۸۰٪ | ۸۸.۷۲٪ | بی‌اثر |
| CIFAR-10 | ۵۲.۸۵٪ | ۵۴.۲۰٪ (`relu(x−k)` با `cov` +3000) | ظاهراً ۱.۳٪ بهتر، فقط ۳ seed |

- انفجارها: روی MNIST ۴۱ بار با `+Cov` در برابر ۷ بار با `−Cov`؛ روی Fashion-MNIST ۴۹ در برابر ۷.
- روی CIFAR-10، در `k+relu` و `k+leaky` جهت منفی در هر ۵ روش بهتر بود؛ در `relu(x−k)` جهت مثبت (چون k با علامت منفی وارد فرمول می‌شود). یعنی در هر دو، جابه‌جایی = −s·Cov.
- Softplus با جهت منفی روی هر سه دیتاست ۱ تا ۶ درصد بهتر شد، ولی هنوز از ReLU ضعیف‌تر ماند.

## آزمایش تأیید روی CIFAR-10 (۱۰ مدل × ۱۰ seed × ۳۰ epoch)

قانون کوواریانس با یادگیری k با گرادیان برابر بود؛ بهبود ۱.۳٪ قبلی تکرار نشد.

| مدل | دقت آخرین epoch (میانگین ± انحراف معیار) |
|---|---|
| SiLU + bias | ۵۷.۰۱٪ ± ۰.۶۳ |
| SiLU با کوواریانس (`cov −100`) | ۵۷.۰۰٪ ± ۰.۵۴ |
| `k+relu` با کوواریانس (`cov_norm −1`) | ۵۵.۱۳٪ ± ۰.۷۳ |
| `relu(x−k)` با گرادیان (`learned`) | ۵۵.۰۳٪ ± ۰.۴۵ |
| `relu(x−k)` با کوواریانس (`cov_tanh +1e4`) | ۵۵.۰۱٪ ± ۰.۶۹ |
| `k+relu` با گرادیان (`learned`) | ۵۵.۰۰٪ ± ۰.۵۱ |
| `k+relu` با کوواریانس (`cov_ema −1`) | ۵۴.۹۳٪ ± ۰.۷۱ |
| `relu(x−k)` با کوواریانس (`cov +3000`) | ۵۴.۸۷٪ ± ۱.۲۳ |
| ReLU بدون bias | ۵۴.۵۴٪ ± ۰.۸۱ |
| ReLU + bias (مرجع) | ۵۴.۳۹٪ ± ۰.۶۱ |

- اختلاف کوواریانس با `learned` همان فرمول: صفر تا ۰.۱۳٪، در حد نوسان تصادفی.
- مرجع بیشتر از بقیه overfit شد (از ۵۵.۳۹٪ در بهترین epoch به ۵۴.۳۹٪)؛ برتری بقیه بر مرجع بیشتر از همین می‌آید.
- با ۹ مقایسه، p حدود ۰.۰۲ تا ۰.۰۵ قابل اعتماد نیست.

## تحلیل

1. در `k+relu` و `relu(x−k)`، bias لایه‌ی کناری می‌تواند جای k را بگیرد؛ پس k توانایی جدیدی اضافه نمی‌کند و تنها اثر ممکنش از نحوه‌ی آموزش است. آزمایش تأیید نشان داد کوواریانس در این زمینه هم از گرادیان بهتر نیست.
2. جهت حرکت k مهم است: منفیِ کوواریانس همیشه پایدارتر بود؛ جهت مثبت چند برابر بیشتر منفجر شد.
3. MNIST و Fashion-MNIST به سقف دقت می‌رسند و تفاوتی نشان نمی‌دهند.
4. بهبود ظاهری ۱.۳٪ روی CIFAR-10 از کم بودن seedها و انتخاب بهترین از بین حدود ۳۳ حالت آمده بود.

## نتیجه‌ی نهایی

- ایده تأیید نشد: قانون کوواریانس بهتر از یادگیری عادی k با گرادیان نیست.
- یافته‌ی جانبی: اگر k با کوواریانس آپدیت شود، فقط جهت منفی پایدار است.
- یافته‌ی جانبی: SiLU روی CIFAR-10 حدود ۲.۵٪ بهتر از ReLU بود، ولی این ربطی به k ندارد.

## لایسنس

MIT — فایل [LICENSE](LICENSE) را ببینید.

</div>
