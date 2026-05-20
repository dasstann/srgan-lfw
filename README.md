# SRGAN: Super-Resolution для лиц на датасете LFW

Финальный проект по дисциплине **Generative Adversarial Networks (GANs)**.
Реализация и обучение генеративно-состязательной сети (SRGAN) для четырёхкратного увеличения разрешения изображений лиц.

**Автор:** Жалмуханбетов Д. Н.
**Преподаватель:** Сапакова С. З.
**Тип проекта:** Super Resolution GAN
**Стек:** Python 3.10, PyTorch 2.x, torchvision, scikit-image, scikit-learn

---

## 1. Название проекта

**SRGAN-LFW: Photo-Realistic 4× Super-Resolution для лиц на датасете Labeled Faces in the Wild**

## 2. Цель проекта

Разработать и обучить генеративно-состязательную сеть, способную восстанавливать высокочастотные детали (текстуру кожи, глаза, волосы) на изображениях лиц с низким разрешением. Модель должна повышать разрешение в 4 раза (с `32×32` до `128×128` пикселей) и сохранять фотореалистичность за счёт комбинированной функции потерь — пиксельной (MSE), состязательной (BCE) и перцептивной (VGG19).

Практическая мотивация: качественное восстановление лиц с кадров видеонаблюдения для последующей работы алгоритмов идентификации и систем контроля посещаемости занятий.

## 3. Описание архитектуры GAN

Проект реализует классическую двухсетевую SRGAN-архитектуру (Ledig et al., 2017) с добавленным perceptual loss и регуляризацией дискриминатора.

### Generator (SRResNet-style)

| Блок | Описание |
|---|---|
| Initial | `Conv2d(3 → 64, 9×9)` + `PReLU` |
| Backbone | **5 × Residual Block** = `Conv3×3 → BN → PReLU → Conv3×3 → BN`, skip-connection |
| Mid | `Conv3×3` + `BN`, skip от Initial |
| Upsample ×2 | `Conv3×3(64 → 256)` + `PixelShuffle(2)` + `PReLU` |
| Upsample ×4 | `Conv3×3(64 → 256)` + `PixelShuffle(2)` + `PReLU` |
| Output | `Conv9×9(64 → 3)` + `Tanh` → диапазон `[-1, 1]` |

Параметры: **~734 тысячи**.

### Discriminator (VGG-style + Dropout)

Каскад из семи свёрток `3×3` с увеличением каналов `64 → 64 → 128 → 128 → 256 → 256 → 512 → 512` и страйдом `2` через блок. В каждом блоке: `Conv → BN → LeakyReLU(0.2) → Dropout2d(0.3)`. Заканчивается `AdaptiveAvgPool(1)`, двумя `1×1`-свёртками и `Sigmoid`.

### VGG19 Feature Extractor (для Perceptual Loss)

Используется VGG19, предобученный на ImageNet, обрезанный до 35-го слоя (`relu5_4`). Веса заморожены. Перед прогоном изображения нормализуются по ImageNet-статистике (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

### Функция потерь генератора

```
L_G = L_pixel + 1e-3 · L_adversarial + 6e-3 · L_perceptual

  L_pixel       = MSE(G(LR), HR)
  L_adversarial = BCE(D(G(LR)), 1)
  L_perceptual  = MSE(φ(G(LR)), φ(HR)),  φ = VGG19/relu5_4
```

### Стратегия обучения

1. **Pre-training (10 эпох):** только `L_pixel`. Цель — научить генератор правильной геометрии и цветовой палитре до подключения adversarial-сигнала.
2. **GAN training (40 эпох):** полный лосс. Применяются Label Smoothing (`0.9` для real), Dropout в D, балансировка весов компонентов лосса.

## 4. Используемый датасет

**Labeled Faces in the Wild (LFW), версия deepfunneled.**

| Параметр | Значение |
|---|---|
| Размер | 13 233 цветных RGB-изображения |
| Уникальных людей | 5 749 |
| Источник | `sklearn.datasets.fetch_lfw_people(color=True)` |
| Train / Test split | 80% / 20% (фиксированный seed=42) |
| Train size | 10 586 изображений |
| Test size | 2 647 изображений |
| HR (целевое) | `128×128`, нормализация в `[-1, 1]` |
| LR (вход) | `32×32`, бикубическая интерполяция, диапазон `[0, 1]` |
| Препроцессинг | CenterCrop 150 → Resize → ToTensor → Normalize |

## 5. Инструкция по запуску

### Вариант A. Google Colab (рекомендуется)

1. Открыть `SRGAN_Final.ipynb` в Google Colab.
2. Среда выполнения → Сменить среду выполнения → **GPU (T4 или лучше)**.
3. Run All. При первом запуске Colab попросит подтверждение на mount Google Drive — нажать «Подключить».
4. После завершения обучения все артефакты будут лежать в `/content/drive/MyDrive/SRGAN_Final/`:
   - `checkpoints/` — веса по эпохам (`*.pth`);
   - `models/generator_final.pth` и `models/discriminator_final.pth` — финальные веса;
   - `results/images/` — `dataset_samples.png`, `bicubic_vs_srgan.png`;
   - `results/plots/` — `training_curves.png`, `epochs_progression.png`;
   - `results/progression/` — `epoch_001.png` ... `epoch_050.png` (по одному PNG на эпоху);
   - `training_history.json` — все loss и метрики;
   - `final_metrics.json` — итоговые PSNR/SSIM.

### Вариант B. Локально

```bash
# 1. Клонировать репозиторий и установить зависимости
git clone <repo_url>
cd srgan-lfw
pip install -r requirements.txt

# 2. Запустить ноутбук
jupyter notebook SRGAN_Final.ipynb
```

При локальном запуске блок mount Google Drive автоматически перейдёт в except-ветку и сохранит артефакты в `./SRGAN_Final/`.

### Минимальные системные требования

- Python 3.10+
- NVIDIA GPU с ≥ 8 GB VRAM (T4, RTX 2070 и выше). На CPU обучение займёт > 24 часов.
- ~3 GB свободного места (датасет + чекпоинты).

## 6. Параметры обучения

| Параметр | Значение | Комментарий |
|---|---|---|
| Optimizer | `Adam` | `betas=(0.9, 0.999)` |
| Learning rate | `1e-4` | Одинаковый для G и D |
| Batch size | `16` | Подобран под 12 GB VRAM (Colab T4) |
| Pre-train epochs | `10` | Только MSE — стабилизация геометрии |
| GAN epochs | `40` | MSE + Adv + Perceptual |
| Adversarial weight | `1e-3` | Против доминирования дискриминатора |
| Perceptual weight | `6e-3` | VGG19/relu5_4, как в оригинальной статье |
| Label smoothing | `0.9` | Для real-меток |
| Dropout | `0.3` | В каждом блоке Discriminator |
| Seed | `42` | `random`, `numpy`, `torch`, `cudnn` |
| Checkpoint frequency | каждые 10 эпох | Защита от обрыва Colab-сессии |
| Evaluation batches | 30 первых батчей test'а | Для скорости в процессе обучения |
| Финальная оценка | весь test (~2 647 изобр.) | Запускается один раз в конце |

## 7. Результаты генерации

Метрики оцениваются на **полном test-сете LFW** (2 647 изображений) после 10 эпох pre-train + 40 эпох GAN-обучения.

| Метод | PSNR (dB) ↑ | SSIM ↑ | Δ к baseline |
|---|---|---|---|
| Bicubic baseline | 29.24 | 0.8705 | — |
| **SRGAN (наша реализация)** | **32.33** | **0.9205** | **+3.09 dB / +0.0500** |

Полные итоговые значения автоматически сохраняются в `final_metrics.json` после прогона ноутбука.

**Динамика обучения (test-сет):**

| Эпоха | Фаза | Test PSNR (dB) | Test SSIM |
|---|---|---|---|
| 1 | Pretrain | 24.77 | 0.8093 |
| 10 | Pretrain | 30.29 | 0.8958 |
| 11 | GAN start | 30.86 | 0.8982 |
| 30 | GAN | 31.93 | 0.9144 |
| 50 | GAN final | **32.29** | **0.9197** |

**Качественные наблюдения:**
- Bicubic-апскейл даёт чёткую геометрию, но сглаживает все текстуры — кожа выглядит «пластиковой».
- SRGAN восстанавливает текстуру кожи, чётче рисует границы глаз и волос. На отдельных изображениях прирост PSNR составляет от **+1.6 до +4.1 dB** относительно Bicubic.
- D loss стабильно держится в коридоре **0.688–0.690** на протяжении всех 40 GAN-эпох — это математическое равновесие Nash, означающее, что дискриминатор не доминирует и нет mode collapse.
- Perceptual loss достигает пика 0.10 при старте GAN-фазы и плавно опускается до 0.06, что коррелирует с улучшением визуальной резкости.

## 8. Скриншоты / примеры результатов

После прогона все артефакты сохраняются автоматически. Ключевые визуализации:

| Файл | Описание |
|---|---|
| `results/images/dataset_samples.png` | 5 пар LR (32×32) / HR (128×128) из датасета |
| `results/images/bicubic_vs_srgan.png` | Сравнение `LR / Bicubic / SRGAN / GT` с PSNR/SSIM под каждым изображением |
| `results/plots/training_curves.png` | Loss curves (G total, D, pixel, perceptual) + PSNR/SSIM по эпохам |
| `results/plots/epochs_progression.png` | Прогрессия генерации одного и того же лица на эпохах 1 / 5 / 10 / 20 / 30 / 40 / 50 |
| `results/progression/epoch_XXX.png` | Полная сетка `LR / Generated / GT` для 8 фиксированных лиц после каждой эпохи |

---

## Структура проекта

```
srgan-lfw/
├── SRGAN_Final.ipynb       # основной ноутбук (обучение + визуализация)
├── README.md               # этот файл
├── requirements.txt        # зависимости
├── SRGAN_Final/            # создаётся ноутбуком
│   ├── checkpoints/        # *.pth по эпохам
│   ├── models/             # generator_final.pth, discriminator_final.pth
│   ├── results/
│   │   ├── images/         # demo-картинки и сравнение Bicubic vs SRGAN
│   │   ├── plots/          # графики обучения
│   │   └── progression/    # snapshot после каждой эпохи
│   ├── training_history.json
│   └── final_metrics.json
└── lfw-deepfunneled/       # скачивается автоматически
```

## Ссылки на литературу

1. Goodfellow, I., et al. *Generative Adversarial Nets.* NIPS, 2014.
2. Ledig, C., et al. *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.* CVPR, 2017.
3. Dong, C., et al. *Image Super-Resolution Using Deep Convolutional Networks.* IEEE TPAMI, 2014.
4. Radford, A., et al. *Unsupervised Representation Learning with Deep Convolutional GANs.* ICLR, 2016.
5. He, K., et al. *Deep Residual Learning for Image Recognition.* CVPR, 2016.
6. Wang, X., et al. *ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.* ECCV Workshops, 2018.
7. Zhang, R., et al. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR, 2018.

## Future Work

- Замена Residual Blocks на RRDB (Residual-in-Residual Dense Block) — переход к архитектуре ESRGAN.
- Relativistic Discriminator (RaGAN).
- Wasserstein loss с gradient penalty (WGAN-GP) как альтернативная стратегия стабилизации.
- Обучение на CelebA-HQ для лиц высокого качества.
- Quantization / ONNX-export для деплоя в продакшен.

## Лицензия

MIT (свободное использование, в т.ч. в академических работах).
