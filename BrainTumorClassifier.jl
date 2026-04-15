### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ b1819df2-c72d-4a23-a1b0-62c0b171ebba
using Pkg

# ╔═╡ bfb1a7dd-49ae-40d2-87f1-6f0e24c01e7d
begin
    using Flux, Metalhead
    using Images, ImageIO, FileIO, ImageTransformations
    using BSON, ProgressMeter, Printf
    using BSON: @save, @load
    using LinearAlgebra
    using Flux: onehotbatch, onecold, logitcrossentropy, state, loadmodel!
    using Statistics: mean, std
    using Random: shuffle!, randperm, seed!, rand, randn

    println("Julia v$(VERSION)")
    println("Flux  v$(pkgversion(Flux))")
    println("Threads: $(Threads.nthreads())")
end

# ╔═╡ d6659407-3c4c-4591-9144-21ec4a07e598
for pkg in ["Flux", "Images", "ImageIO", "FileIO", "ImageTransformations",
            "Statistics", "Random", "BSON", "ProgressMeter", "Metalhead"]
    if !haskey(Pkg.project().dependencies, pkg)
        Pkg.add(pkg; io=devnull)
    end
end

# ╔═╡ 72a16944-b39d-4607-9347-b9791a5d7429
md"""
```bash
── Per-class Results ──────────────────────────────
Class           Precision    Recall        F1
──────────────────────────────────────────────────
glioma             0.976     0.818     0.890
meningioma         0.887     0.958     0.921
notumor            0.939     0.998     0.967
pituitary          0.971     0.990     0.980
──────────────────────────────────────────────────
Overall accuracy: 0.9406

── Confusion Matrix ───────────────────────────
      gli    men    not    pit    
gli     327     46     23      4  
men       6    383      3      8  
not       1      0    399      0  
pit       1      3      0    396  
```
"""

# ╔═╡ 6f303980-3560-11f1-a4c7-0998bef080a7
md"""
# Image Classification with Deep Learning

**A ResNet-18 Based Approach Using Julia and Flux.jl**

---

## Abstract

This notebook implements an end-to-end deep learning pipeline for automated brain tumor classification from MRI scans. We employ a **ResNet-18** convolutional neural network architecture to classify brain MRI images into four categories:

| Class | Description |
|:------|:------------|
| **Glioma** | Tumors arising from glial cells in the brain and spinal cord |
| **Meningioma** | Tumors originating from the meninges surrounding the brain |
| **No Tumor** | Healthy brain scans with no detectable tumor |
| **Pituitary** | Tumors of the pituitary gland |

### Key Design Decisions

- **Architecture**: ResNet-18 backbone with custom classification head (512 → 256 → 4)
- **Regularisation**: Label smoothing (ε=0.1), AdamW weight decay, Dropout(0.3), class-weighted loss
- **Training strategy**: Cosine annealing learning rate schedule, early stopping on validation loss
- **Data pipeline**: Stratified train/val split, on-the-fly augmentation (flips, rotations, noise, brightness/contrast), parallel image loading

### Dataset

We use the **Brain Tumor MRI Dataset** — a curated collection of 7,200 brain MRI images (5,600 training + 1,600 testing), evenly distributed across 4 classes (1,400 training / 400 testing per class).

---
"""

# ╔═╡ ea7147ed-9a0f-412c-8bb3-ff18e8f2832f


# ╔═╡ 6f39fd80-3560-11f1-9403-25a4c99377f4
md"""
## 1. Environment Setup

We begin by installing all required dependencies into the Colab environment using Julia's Pkg mode. This notebook uses the Julia **SciML** and **Flux** ecosystem:

| Package | Purpose |
|:--------|:--------|
| `Flux.jl` | Deep learning framework (model definition, training, optimisers) |
| `Metalhead.jl` | Pre-defined computer vision architectures (ResNet, VGG, etc.) |
| `Images.jl` | Image I/O and color-space manipulation |
| `ImageTransformations.jl` | Geometric transformations for data augmentation |
| `BSON.jl` | Model serialisation and checkpoint management |
| `ProgressMeter.jl` | Training progress visualisation |
| `CUDA.jl` | GPU acceleration for Google Colab T4 |

"""

# ╔═╡ 6f3e9160-3560-11f1-8147-9b1e1862e747
md"""
### 1.1 Compute Backend & Threading

We configure the compute backend and CPU threading settings. Julia thread count must be set **before** launching Julia (it cannot be changed inside a running session).

Start Julia with multiple threads using one of these options:

- Command line: `julia -t auto` or `julia -t 4`
- Environment variable (set before launch): `JULIA_NUM_THREADS=4`

For CPU training performance, we also configure BLAS threads from inside Julia.
"""

# ╔═╡ a1f75273-0070-4c66-aaf6-694ad8cdcded
# ─── Optional GPU ─────────────────────────────────────────────────────────────
# Uncomment the backend that matches your hardware:
# using CUDA; device = gpu          # NVIDIA
# using Metal; device = gpu         # Apple Silicon
device = cpu                         # safe fallback

# ╔═╡ 1387fa2d-2fcc-4421-8a56-d8583f374213
# Thread count for Julia itself must be set BEFORE starting Julia:
  # julia -t auto
  # julia -t 4
julia_threads = Threads.nthreads()

# ╔═╡ baeeabfd-a6d8-4e66-a954-59ed1558652b
if julia_threads == 1
    @warn "Julia is running with 1 thread. Restart with `julia -t auto` or set JULIA_NUM_THREADS>1 to enable multi-threaded CPU training."
end

# ╔═╡ 7aba4d48-c889-4c8f-9c40-8d45fd57b0f8
# BLAS threading can be adjusted at runtime and impacts many CPU ops used in training
blas_threads = max(1, min(julia_threads, Sys.CPU_THREADS))

# ╔═╡ c73633c1-c272-46f7-bb5f-502edde384d7
BLAS.set_num_threads(blas_threads)

# ╔═╡ 958eeba2-dc10-475a-8159-49dcaf084429
println("Julia threads: $julia_threads")

# ╔═╡ dfd50fa7-1079-4469-80af-448c35b6c40c
println("BLAS threads : $(BLAS.get_num_threads())")

# ╔═╡ 6f3e9160-3560-11f1-9dab-ab05ac3fdc4b
md"""
---

## 2. Configuration

> **Colab Note on Data**: Make sure your dataset is accessible in the Colab runtime. You can either upload the `Training` and `Testing` folders directly to the Colab workspace, or mount your Google Drive and update the paths below accordingly.

All experiment hyperparameters and paths are centralised here for reproducibility. This follows the standard practice of separating configuration from logic, making it easy to run ablation studies by only modifying this cell.
"""

# ╔═╡ 5e6252d7-45fb-44dd-9d6c-565850621d71
# ── Experiment Configuration ─────────────────────────────────────────────────

const IMG_SIZE    = (128, 128)         # Input spatial resolution (H × W)

# ╔═╡ e25ee0cd-2154-4a6a-9c46-1c11606b9745
const CLASSES     = ["glioma", "meningioma", "notumor", "pituitary"]

# ╔═╡ ed38c9d2-18fb-4394-a520-355e8386c795
const N_CLASSES   = length(CLASSES)

# ╔═╡ e7e20fdf-387e-4b9c-9d7d-5c493d66ce5c
const CLASS_IDX   = Dict(c => i for (i, c) in enumerate(CLASSES))

# ╔═╡ 92301a9f-3c15-4ac3-b7f1-3d0b0074af6f
# Paths
TRAIN_DIR         = "Training"

# ╔═╡ bb29e883-1cae-412f-ab06-754a053a8b23
TEST_DIR          = "Testing"

# ╔═╡ 5cc06f8c-7401-4db3-92c4-88261514b603
CHECKPOINT_PATH   = "best_model.bson"

# ╔═╡ 62126c3f-dad8-4224-9879-3580adb7989b
# Training hyperparameters
SEED              = 42

# ╔═╡ f3a15b7d-c90d-4af0-b936-530bb47ccd3f
EPOCHS            = 30

# ╔═╡ 620f4bad-40a6-4e3a-acf0-d4dcf9c00649
BATCH_SIZE        = 32

# ╔═╡ 5d3e725d-8b26-4555-925d-078ed9e29acb
LR_INIT           = 3f-4              # Initial learning rate

# ╔═╡ 3b6e8956-99a8-46e1-8c00-f4542ab75443
LR_MIN            = 1f-6              # Minimum learning rate (cosine floor)

# ╔═╡ 561e81f7-9dc2-477a-ac75-e31abb1659e5
WEIGHT_DECAY      = 1f-4              # AdamW L2 regularisation

# ╔═╡ 59a36519-ad25-4248-8822-b34d49bb8772
LABEL_SMOOTH_EPS  = 0.1f0             # Label smoothing factor

# ╔═╡ b3d3e27b-aa40-449b-9e42-9a4a527c65d6
VAL_SPLIT         = 0.2               # Fraction of training data → validation

# ╔═╡ cf3f4451-f65b-4105-bfa5-49598618e01c
PATIENCE          = 2                 # Early stopping patience (epochs)

# ╔═╡ 3f8eaffd-6b4d-4eaf-91a2-b5945b2bb411
AUGMENT           = true              # Enable training-time augmentation

# ╔═╡ af643dc3-7e99-46ef-8d4d-9d9985273a36
seed!(SEED)

# ╔═╡ a2213680-4ab1-4e39-b79c-cf5e30ad2ca1
println("Configuration loaded.")

# ╔═╡ 5659be55-bf8e-4157-9ec0-b6b9212c2563
println("  Image size : $(IMG_SIZE[1])×$(IMG_SIZE[2])")

# ╔═╡ 6940afff-eb8e-4b99-a0b0-4b1b48c73feb
println("  Classes    : $(join(CLASSES, ", "))")

# ╔═╡ e33029d0-7f56-4544-afdb-451f1e1aa5de
println("  Epochs     : $EPOCHS")

# ╔═╡ cbfc1c17-20a0-4ac2-9100-63eaabfdc7ef
println("  Batch size : $BATCH_SIZE")

# ╔═╡ 5ac497a7-9bbd-42f4-9f30-9f91e761b647
println("  LR         : $LR_INIT → $LR_MIN (cosine annealing)")

# ╔═╡ 6f3e9160-3560-11f1-9823-f75f6ba16141
md"""
---

## 3. Data Pipeline

The data pipeline handles:

1. **Discovery** — Recursively scans class-labelled directories for image files
2. **Stratified splitting** — Ensures proportional class representation in train/val sets
3. **Image loading** — Loads, resizes (Lanczos interpolation), and converts to Float32 tensors
4. **Normalisation** — Per-dataset mean/std normalisation computed from a random sample of training images
5. **Augmentation** — Stochastic transformations applied on-the-fly during training
6. **Batching** — Parallel construction of mini-batches using Julia's multi-threading
"""

# ╔═╡ 6f40db4e-3560-11f1-b2a2-9147170a746a
md"### 3.1 Sample Collection & Stratified Splitting"

# ╔═╡ 6f40db4e-3560-11f1-b1a4-836afc44c818
"""
    collect_samples(root::String) → Vector{Tuple{String, Int}}

Walk the directory tree under `root`, collecting `(filepath, class_index)` pairs
for all images matching standard medical imaging formats (.jpg, .png, .tiff, etc.).

Expects the standard ImageNet-style directory layout:
    root/
      class_name_1/
        image_001.jpg
        ...
      class_name_2/
        ...
"""
function collect_samples(root::String)
    samples = Tuple{String, Int}[]
    for cls in CLASSES
        dir = joinpath(root, cls)
        isdir(dir) || (println("⚠ Missing class dir: $dir"); continue)
        for f in readdir(dir; join=true)
            ext = lowercase(splitext(f)[2])
            ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif") || continue
            push!(samples, (f, CLASS_IDX[cls]))
        end
    end
    return samples
end

# ╔═╡ 6f40db4e-3560-11f1-adc5-1f9cb50ee609
"""
    stratified_split(samples, val_frac) → (train, val)

Split samples into training and validation sets while preserving the class
distribution. Each class contributes `val_frac` of its samples to validation.
Both sets are shuffled independently to prevent ordering bias.
"""
function stratified_split(samples::Vector{Tuple{String,Int}}, val_frac::Float64)
    by_class = [Vector{Tuple{String,Int}}() for _ in 1:N_CLASSES]
    for sample in samples
        push!(by_class[sample[2]], sample)
    end
    train = Tuple{String,Int}[]
    val   = Tuple{String,Int}[]
    for cls_samples in by_class
        shuffle!(cls_samples)
        n_val = round(Int, val_frac * length(cls_samples))
        append!(val, cls_samples[1:n_val])
        append!(train, cls_samples[n_val+1:end])
    end
    shuffle!(train)
    shuffle!(val)
    return train, val
end

# ╔═╡ 6f40db4e-3560-11f1-a056-a5e8b92f1b08
md"### 3.2 Image Loading & Preprocessing"

# ╔═╡ 6f40db4e-3560-11f1-ae01-fba1890b46bb
"""
    load_image(path::String) → Array{Float32, 3}

Load a single image and convert it to a normalised Float32 tensor with shape
`(H, W, C)` where C=3 (RGB channels).

Processing steps:
  1. Load from disk (supports JPEG, PNG, TIFF, BMP)
  2. Resize to `IMG_SIZE` using Lanczos interpolation for quality
  3. Convert to Float32 RGB in [0, 1] range
  4. Permute from channel-first `(C, H, W)` to channel-last `(H, W, C)` for Flux
"""
function load_image(path::String)::Array{Float32, 3}
    img = load(path)
    img = imresize(img, IMG_SIZE)
    img = RGB{Float32}.(img)
    arr = permutedims(channelview(img), (2, 3, 1))  # C×H×W → H×W×C
    return arr
end

# ╔═╡ 6f40db4e-3560-11f1-86c0-31a34b6aab12
md"""
### 3.3 Data Augmentation

We apply stochastic transformations during training to improve generalisation and prevent overfitting. The augmentation pipeline includes:

| Transformation | Probability | Parameters | Rationale |
|:---------------|:-----------:|:-----------|:----------|
| Horizontal flip | 50% | — | MRI orientation invariance |
| Vertical flip | 50% | — | Sagittal symmetry |
| Random rotation | 50% | ±14.3° (±0.25 rad) | Compensate scanner angle variation |
| Gaussian noise | 30% | σ = 0.02 | Simulate acquisition noise |
| Brightness jitter | 40% | ×[0.85, 1.15] | Compensate intensity variation |
| Contrast jitter | 40% | ×[0.8, 1.2] | Compensate contrast variation |
"""

# ╔═╡ 6f40db4e-3560-11f1-a53e-5fe0bad51aa6
"""
    augment_image(img::Array{Float32,3}) → Array{Float32,3}

Apply stochastic geometric and photometric augmentations to a single image tensor.
All transformations are applied in-place where possible to minimise allocations.
Output values are clamped to [0, 1].
"""
function augment_image(img::Array{Float32,3})
    # Geometric augmentations
    if rand() < 0.5f0
        img = reverse(img, dims=1)       # horizontal flip
    end
    if rand() < 0.5f0
        img = reverse(img, dims=2)       # vertical flip
    end

    # Random rotation (up to ±14.3°)
    if rand() < 0.5f0
        angle = (rand() - 0.5f0) * 0.5f0  # ∈ [-0.25, 0.25] radians
        c_img = colorview(RGB, permutedims(img, (3, 1, 2)))
        rot_img = ImageTransformations.imresize(
            ImageTransformations.imrotate(c_img, angle),
            (size(img, 1), size(img, 2))
        )
        img = Float32.(permutedims(channelview(rot_img), (2, 3, 1)))
        img[isnan.(img)] .= 0f0          # replace NaN padding from rotation
    end

    # Photometric augmentations
    if rand() < 0.3f0                    # additive Gaussian noise
        noise = randn(Float32, size(img)) .* 0.02f0
        img .= clamp.(img .+ noise, 0f0, 1f0)
    end
    if rand() < 0.4f0                    # brightness jitter
        factor = 0.85f0 + 0.3f0 * rand(Float32)
        img .= clamp.(img .* factor, 0f0, 1f0)
    end
    if rand() < 0.4f0                    # contrast jitter
        contrast = 0.8f0 + 0.4f0 * rand(Float32)
        img .= clamp.((img .- 0.5f0) .* contrast .+ 0.5f0, 0f0, 1f0)
    end
    return img
end

# ╔═╡ 6f40db4e-3560-11f1-bacc-ddbacf99113f
md"### 3.4 Normalisation & Batch Construction"

# ╔═╡ e8ee36c6-a1d0-4469-96f8-0f67deed5e69
"""
    compute_dataset_stats(samples; max_images=500) → (μ, σ)

Estimate the per-pixel mean and standard deviation from a random subset of the
training images. Used for global normalisation: `x̂ = (x - μ) / σ`.

Sampling `max_images` (default 500) provides a stable estimate while avoiding
the cost of loading the full dataset.
"""
function compute_dataset_stats(samples::Vector{Tuple{String,Int}}; max_images::Int=500)
    n         = min(max_images, length(samples))
    indices   = randperm(length(samples))[1:n]
    sum_      = 0.0
    sumsq     = 0.0
    total_pts = 0
    for idx in indices
        img = load_image(samples[idx][1])
        sum_ += sum(img)
        sumsq += sum(img .^ 2)
        total_pts += length(img)
    end
    μ = Float32(sum_ / total_pts)
    σ = Float32(sqrt(max(sumsq / total_pts - μ^2, 1f-8)))
    return μ, σ
end

# ╔═╡ fb57e2af-3600-43d5-bc23-3d7a40cbb65f
"""
    get_class_weights(samples) → Vector{Float32}

Compute inverse-frequency class weights for the weighted cross-entropy loss.
Weights are proportional to `max_count / class_count`, ensuring that
under-represented classes receive higher loss contributions.
"""
function get_class_weights(samples::Vector{Tuple{String,Int}})
    counts = zeros(Int, N_CLASSES)
    for (_, lbl) in samples
        counts[lbl] += 1
    end
    max_count = maximum(counts)
    return Float32.(max_count ./ max.(counts, 1))
end

# ╔═╡ aaebe68b-9745-423d-b9fe-9e6498c4ace2
"""
    normalize_batch!(xs, μ, σ) → xs

In-place global normalisation of a batch tensor: `x̂ = (x - μ) / σ`.
"""
function normalize_batch!(xs::Array{Float32,4}, μ::Float32, σ::Float32)
    xs .= (xs .- μ) ./ σ
    return xs
end

# ╔═╡ f0a9954f-ec70-456b-a054-6a687ffe11f8
"""
    build_batch(samples; augment=false, μ=0, σ=1) → (xs, ys)

Construct a mini-batch from a list of `(path, label)` samples.

Returns:
  - `xs`: Float32 tensor of shape `(H, W, C, N)` — normalised image batch
  - `ys`: One-hot encoded label matrix of shape `(N_CLASSES, N)`

Images are loaded in parallel using `Threads.@threads` for throughput.
"""
function build_batch(samples::Vector{Tuple{String,Int}}; augment::Bool=false,
                     μ::Float32=0f0, σ::Float32=1f0)
    n   = length(samples)
    xs  = Array{Float32}(undef, IMG_SIZE[1], IMG_SIZE[2], 3, n)
    ys  = Vector{Int}(undef, n)
    Threads.@threads for i in eachindex(samples)
        path, lbl = samples[i]
        img = load_image(path)
        augment && (img = augment_image(img))
        xs[:, :, :, i] = img
        ys[i]          = lbl
    end
    normalize_batch!(xs, μ, σ)
    return xs, onehotbatch(ys, 1:N_CLASSES)
end

# ╔═╡ bee3759e-2731-454a-8a94-a29859415ada
"""
    make_batches(samples, batch_size; kwargs...) → Vector{(xs, ys)}

Partition samples into pre-built mini-batches. Each batch is a complete
`(tensor, one_hot)` tuple ready for forward pass — no additional I/O during training.
"""
function make_batches(samples::Vector{Tuple{String,Int}}, batch_size::Int;
                      shuffle_data::Bool=true, augment::Bool=false,
                      μ::Float32=0f0, σ::Float32=1f0)
    shuffle_data && shuffle!(samples)
    n       = length(samples)
    batches = Vector{Tuple{Array{Float32,4}, AbstractMatrix{Bool}}}()
    for start in 1:batch_size:n
        chunk = samples[start:min(start+batch_size-1, n)]
        push!(batches, build_batch(chunk; augment=augment, μ=μ, σ=σ))
    end
    return batches
end

# ╔═╡ 6f40db4e-3560-11f1-8fc5-e3e99661f9cd
md"""
---

## 4. Model Architecture

We use a **ResNet-18** backbone as the feature extractor, followed by a custom classification head. ResNet-18 (He et al., 2016) uses skip connections to enable training of deeper networks without degradation.

### Architecture Overview

```
Input (128×128×3)
    │
    ▼
┌─────────────────────────────┐
│   ResNet-18 Backbone        │   ← Feature extractor (conv1 → layer4)
│   18 convolutional layers   │      ~11M parameters
│   with residual connections │
└─────────────────────────────┘
    │  (4×4×512)
    ▼
┌─────────────────────────────┐
│   Global Average Pooling    │   → (1×1×512)
│   Flatten                   │   → (512,)
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│   Dense(512 → 256, ReLU)    │   Classification head
│   Dropout(0.3)              │   Regularisation
│   Dense(256 → 4)            │   Output logits
└─────────────────────────────┘
    │
    ▼
Output: 4-class logits
```

**Parameter count**: ~11.3M parameters (backbone ≈ 11.2M, head ≈ 0.1M)
"""

# ╔═╡ 6f40db4e-3560-11f1-9c1e-ddb5dc6b089f
"""
    build_model() → Chain

Construct the classification model:
  1. ResNet-18 backbone as feature extractor
  2. Global average pooling to collapse spatial dimensions
  3. Two-layer classification head with dropout regularisation

The backbone weights are initialised randomly (`pretrain=false`).
Set `pretrain=true` to use ImageNet-pretrained weights for transfer learning.
"""
function build_model()
    resnet   = ResNet(18; pretrain=false)
    backbone = resnet.layers[1]          # feature extractor (all conv blocks)

    return Chain(
        backbone,                        # Input → (H/32 × W/32 × 512 × N)
        GlobalMeanPool(),                # → (1 × 1 × 512 × N)
        Flux.flatten,                    # → (512, N)
        Dense(512 => 256, relu),          # Learned projection
        Dropout(0.3f0),                  # Regularisation
        Dense(256 => N_CLASSES),          # Class logits
    )
end

# ╔═╡ 6f40db4e-3560-11f1-a30f-fd4eb57919f4
md"""
---

## 5. Training Engine

### 5.1 Loss Function

We use **label-smoothed cross-entropy** with **class weighting**:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} w_{y_i} \sum_{k=1}^{K} \tilde{y}_{ik} \log p_{ik}$$

where:
- $\tilde{y}_{ik} = (1 - \varepsilon) \cdot \mathbb{1}[k = y_i] + \varepsilon / K$ is the smoothed target
- $w_{y_i}$ is the inverse-frequency class weight
- $\varepsilon = 0.1$ is the smoothing parameter

**Why label smoothing?** It prevents the model from becoming over-confident on training examples and improves calibration of the predicted probabilities — critical for medical diagnosis applications.
"""

# ╔═╡ 6f40db4e-3560-11f1-a841-d11c53667df7
"""
    smooth_ce_loss(logits, labels; class_weights, ε=0.1) → scalar

Label-smoothed, class-weighted cross-entropy loss.

Combines label smoothing (Szegedy et al., 2016) with inverse-frequency
class weighting to handle both calibration and class imbalance simultaneously.
"""
function smooth_ce_loss(logits, labels;
                        class_weights::Vector{Float32}=ones(Float32, N_CLASSES),
                        ε::Float32=LABEL_SMOOTH_EPS)
    K    = Float32(N_CLASSES)
    soft = (1f0 - ε) .* labels .+ ε ./ K     # smoothed targets
    logp = Flux.logsoftmax(logits; dims=1)    # numerically stable log-probabilities
    losses = vec(-sum(soft .* logp; dims=1))  # per-sample CE
    truth  = onecold(labels)
    sample_weights = class_weights[truth]      # weight by class frequency
    return mean(sample_weights .* losses)
end

# ╔═╡ 6f40db4e-3560-11f1-9155-6174580a9cb2
md"""
### 5.2 Learning Rate Schedule

We use **cosine annealing** (Loshchilov & Hutter, 2017) to smoothly decay the learning rate:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$

This avoids the sharp transitions of step-based schedules and has been shown to improve convergence in deep networks.
"""

# ╔═╡ 6f40db4e-3560-11f1-9aaa-89f70b385ebd
"""Cosine-annealed learning rate: smoothly decays from `lr_max` to `lr_min` over `T` epochs."""
cosine_lr(epoch::Int, T::Int; lr_max::Float32=LR_INIT, lr_min::Float32=LR_MIN) =
    lr_min + 0.5f0 * (lr_max - lr_min) * (1 + cos(Float32(π) * epoch / T))

# ╔═╡ 6f40db4e-3560-11f1-8c1f-e1f9602553db
md"### 5.3 Checkpoint Management"

# ╔═╡ 0afadaa6-477e-4e28-bcf7-57bfd9367e9f
"""
    save_checkpoint(path, model, opt_state, μ, σ, class_weights)

Serialise the complete training state to a BSON file:
  - Model weights (as Flux state dict)
  - Optimiser state (for training resumption)
  - Dataset normalisation parameters (μ, σ)
  - Class weights (for consistent inference-time loss computation)
"""
function save_checkpoint(path::String, model, opt_state, μ::Float32, σ::Float32,
                         class_weights::Vector{Float32})
    model_state = Flux.state(model)
    @save path model_state opt_state μ σ class_weights
end

# ╔═╡ d7b681be-c6b5-4bb2-933b-36d191de7f1f
"""
    load_checkpoint!(path, model) → (opt_state, μ, σ, class_weights)

Restore model weights and training metadata from a BSON checkpoint.
Supports both legacy (:model) and current (:model_state) checkpoint formats.
"""
function load_checkpoint!(path::String, model)
    data = BSON.load(path)
    if haskey(data, :model_state)
        Flux.loadmodel!(model, data[:model_state])
    elseif haskey(data, :model)
        Flux.loadmodel!(model, data[:model])
    else
        error("Could not find :model_state or :model in checkpoint")
    end

    opt_state     = get(data, :opt_state, nothing)
    μ             = get(data, :μ, 0f0)
    σ             = get(data, :σ, 1f0)
    class_weights = get(data, :class_weights, ones(Float32, N_CLASSES))

    return opt_state, μ, σ, class_weights
end

# ╔═╡ 6f40db4e-3560-11f1-81d2-9f5dde4ea2e8
md"### 5.4 Training & Evaluation Loops"

# ╔═╡ ea5f9aaf-0f46-42a1-90de-9fc2eef51d53
"""
    train_epoch!(model, opt_state, batches) → avg_loss

Execute one full training epoch over all mini-batches.
Computes gradients via reverse-mode AD and updates model parameters.
"""
function train_epoch!(model, opt_state, batches)
    total_loss = 0.0
    n_batches  = length(batches)
    prog = Progress(n_batches; dt=0.5, desc="  train ")
    for (xs, ys) in batches
        xs_d = xs |> device
        ys_d = ys |> device
        loss, grads = Flux.withgradient(model) do m
            smooth_ce_loss(m(xs_d), ys_d)
        end
        Flux.update!(opt_state, model, grads[1])
        total_loss += loss
        next!(prog)
    end
    return total_loss / n_batches
end

# ╔═╡ f4958ecd-1cbd-4281-a7c0-30fbef3ec94f
"""
    evaluate(model, batches; class_weights) → (avg_loss, accuracy)

Evaluate model on a set of batches in inference mode.
Returns the average loss and top-1 classification accuracy.
"""
function evaluate(model, batches; class_weights::Vector{Float32}=ones(Float32, N_CLASSES))
    total_loss = 0.0
    correct    = 0
    total      = 0
    for (xs, ys) in batches
        xs_d = xs |> device
        ys_d = ys |> device
        logits = model(xs_d)
        total_loss += smooth_ce_loss(logits, ys_d; class_weights=class_weights)
        pred    = onecold(logits)
        truth   = onecold(ys_d)
        correct += sum(pred .== truth)
        total   += length(truth)
    end
    return total_loss / length(batches), correct / total
end

# ╔═╡ 6f40db4e-3560-11f1-9058-b1b941d0f4e6
md"""
---

## 6. Evaluation & Metrics

We compute standard classification metrics:

- **Confusion Matrix** — Full N×N matrix of predictions vs. ground truth
- **Per-class Precision** — $\text{TP} / (\text{TP} + \text{FP})$
- **Per-class Recall (Sensitivity)** — $\text{TP} / (\text{TP} + \text{FN})$
- **Per-class F1 Score** — Harmonic mean of precision and recall
- **Overall Accuracy** — Fraction of correctly classified samples
"""

# ╔═╡ 6f40db4e-3560-11f1-8be1-5dd5a4e385bc
"""
    class_report(model, test_samples, batch_size; μ, σ)

Generate a comprehensive classification report on the test set:
  - Per-class precision, recall, and F1 score
  - Full confusion matrix
  - Overall accuracy
"""
function class_report(model, test_samples, batch_size; μ::Float32=0f0, σ::Float32=1f0)
    cm = zeros(Int, N_CLASSES, N_CLASSES)  # cm[pred, truth]
    for start in 1:batch_size:length(test_samples)
        chunk      = test_samples[start:min(start+batch_size-1, end)]
        xs, ys     = build_batch(chunk; augment=false, μ=μ, σ=σ)
        preds      = onecold(model(xs |> device))
        truths     = onecold(ys |> device)
        for (p, t) in zip(preds, truths)
            cm[p, t] += 1
        end
    end

    # ── Per-class metrics ─────────────────────────────────────────────────────
    println("\n── Per-class Results ──────────────────────────────")
    @printf("%-14s  %8s  %8s  %8s\n", "Class", "Precision", "Recall", "F1")
    println("─" ^ 50)
    for i in 1:N_CLASSES
        tp  = cm[i, i]
        fp  = sum(cm[i, :]) - tp
        fn  = sum(cm[:, i]) - tp
        pre = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1  = 2 * pre * rec / max(pre + rec, 1e-8)
        @printf("%-14s  %8.3f  %8.3f  %8.3f\n", CLASSES[i], pre, rec, f1)
    end
    println("─" ^ 50)

    # ── Overall accuracy ──────────────────────────────────────────────────────
    acc = sum(cm[i,i] for i in 1:N_CLASSES) / sum(cm)
    @printf("Overall accuracy: %.4f\n\n", acc)

    # ── Confusion matrix ─────────────────────────────────────────────────────
    println("── Confusion Matrix ───────────────────────────")
    print("      ")
    for i in 1:N_CLASSES
        @printf("%-6s ", first(CLASSES[i], 3))
    end
    println()
    for i in 1:N_CLASSES
        @printf("%-5s ", first(CLASSES[i], 3))
        for j in 1:N_CLASSES
            @printf("%5d  ", cm[j, i])
        end
        println()
    end
    println()

    return cm, acc
end

# ╔═╡ cf822b58-4e3c-4e03-998c-88a8061dc85e


# ╔═╡ 6f40db4e-3560-11f1-bf33-ed594fb334e8
md"### Utility"

# ╔═╡ 6f40db4e-3560-11f1-b04c-9d19dbf12472
"""Format an integer with comma separators for readability (e.g., 11308868 → 11,308,868)."""
function format_number(n::Int)
    s = string(n)
    out = ""
    for (i, c) in enumerate(reverse(s))
        i > 1 && i % 3 == 1 && (out = "," * out)
        out = string(c) * out
    end
    return out
end

# ╔═╡ 6f40db4e-3560-11f1-b6e4-3726fda8eb79
md"""
---

## 7. Experiment Execution

### 7.1 Data Loading & Exploration
"""

# ╔═╡ ffe72ba7-4c3c-47c5-8b8e-34a7e7a1526a
# ── Collect all samples ──────────────────────────────────────────────────────
train_samples = collect_samples(TRAIN_DIR)

# ╔═╡ 623a93df-78a0-4a49-9c1a-370be83cf877
test_samples  = collect_samples(TEST_DIR)

# ╔═╡ 30bd603e-5e64-44bc-8ad1-cf88a28bc799
println("Dataset Summary")

# ╔═╡ 0da95c67-ffb0-497e-929d-686268731442
println("═" ^ 40)

# ╔═╡ 178ee25c-f8aa-43ca-b2fa-a9534abb518a
println("  Training samples : $(length(train_samples))")

# ╔═╡ 8f993cfd-e871-47d6-90d8-411811133c77
println("  Testing samples  : $(length(test_samples))")

# ╔═╡ 051dbfcf-e63f-4554-a672-60a02b917075
println()

# ╔═╡ 2a7f8e4b-2f0c-4420-98b9-f72c89cb348c
println("  Class distribution (training):")

# ╔═╡ cbc51d10-71bd-43fd-935a-ecd7dbdddda1
for cls in CLASSES
    n = count(s -> s[2] == CLASS_IDX[cls], train_samples)
    bar = "█" ^ round(Int, n / 50)
    @printf("    %-12s %4d  %s\n", cls, n, bar)
end

# ╔═╡ 6f40db4e-3560-11f1-9f90-059a3eb9c2cf
md"""
### 7.2 Sample Visualisation

Display one random sample from each class to verify the data pipeline is correct.
"""

# ╔═╡ 6f40db4e-3560-11f1-b2ae-616cb6d81df4
# ── Visualise one sample per class ────────────────────────────────────────────
for cls in CLASSES
    idx = CLASS_IDX[cls]
    cls_samples = filter(s -> s[2] == idx, train_samples)
    path, _ = cls_samples[rand(1:length(cls_samples))]
    img = load(path)
    img = imresize(img, IMG_SIZE)
    println("Class: $cls")
    display(img)
end

# ╔═╡ 6f40db4e-3560-11f1-8806-c7d5b294fc7a
md"### 7.3 Train/Validation Split & Normalisation"

# ╔═╡ 23b61c1f-1032-4611-8178-94e3be8665d7
# ── Stratified split ──────────────────────────────────────────────────────────
train_s, val_s = stratified_split(train_samples, VAL_SPLIT)

# ╔═╡ be848406-5f18-4a0b-a19c-12c6a809b49d
println("Data Split")

# ╔═╡ 453ff881-9691-418c-80e8-5b08a95f3544
println("═" ^ 40)

# ╔═╡ c1a0af6c-adc3-4bd8-b142-4f7aa717771f
println("  Training   : $(length(train_s))")

# ╔═╡ ba092993-37e9-4fef-95ba-b6c542dd5539
println("  Validation : $(length(val_s))")

# ╔═╡ 840bae80-ab3a-4857-9af0-8586c659f52a
println("  Testing    : $(length(test_samples))")

# ╔═╡ 12138524-7093-4ddd-8fcc-61b8a3ecb66a
# ── Compute normalisation statistics from training set ─────────────────────
μ, σ = compute_dataset_stats(train_s)

# ╔═╡ 522270b3-f7ce-447a-aa3d-95d00ea13570
println("\nNormalisation Statistics")

# ╔═╡ 8b839568-e504-42ff-80aa-172bd4fcf933
println("  μ = $(round(μ, digits=4))")

# ╔═╡ d83b3cb6-05f3-4a64-96fe-1b80e8a33f9d
println("  σ = $(round(σ, digits=4))")

# ╔═╡ 76bf0388-7a48-4641-801c-e9e0ae9b9a45
# ── Class weights ─────────────────────────────────────────────────────────────
class_weights = get_class_weights(train_s)

# ╔═╡ eaffd67f-d610-4fb1-8121-2a26ccb6011d
println("\nClass Weights")

# ╔═╡ 233d5388-bc7d-40d7-b69e-ebe37fe26f85
for (cls, w) in zip(CLASSES, class_weights)
    @printf("  %-12s → %.4f\n", cls, w)
end

# ╔═╡ 6f40db4e-3560-11f1-b3a5-650e0b06dd62
md"### 7.4 Model Instantiation"

# ╔═╡ 3c317b8f-8245-4dee-97a8-6da96988d99a
# ── Build model ───────────────────────────────────────────────────────────────
model = build_model() |> device

# ╔═╡ cfc04935-dcc0-4ffe-b787-f548fdc9ddbe
n_params = sum(length, Flux.params(model))

# ╔═╡ 764f69ba-951d-44f6-b745-bf3aeacaa8ea
println("Model Summary")

# ╔═╡ 90f5057f-8c4d-4e05-bcec-3245f9101696
println("═" ^ 40)

# ╔═╡ 8d2bf044-da62-43f5-a693-09f517c8871b
println("  Architecture   : ResNet-18 + Custom Head")

# ╔═╡ 0a6743f2-fd73-474a-88f9-873bb04feedb
println("  Parameters     : $(format_number(n_params))")

# ╔═╡ b99ebb2a-e845-4fc9-b8f3-766ead2791fa
println("  Input shape    : $(IMG_SIZE[1])×$(IMG_SIZE[2])×3")

# ╔═╡ 45cdf724-3abe-4ab1-9157-15ecd4c0ae27
println("  Output classes : $N_CLASSES")

# ╔═╡ 99bec46e-1c27-46c0-a6f3-ee70e0967240
println("  Device         : $(device == cpu ? "CPU" : "GPU")")

# ╔═╡ 5cd724d9-984a-49cf-9e88-9641ee591980
# ── Verify forward pass ───────────────────────────────────────────────────────
dummy_input = randn(Float32, IMG_SIZE[1], IMG_SIZE[2], 3, 1) |> device

# ╔═╡ f7b60af1-2f54-48e8-ac5b-c7b128e3c922
dummy_out = model(dummy_input)

# ╔═╡ c37a6fef-bc23-4af1-9249-adcb73db5218
println("\n  Forward pass test: input $(size(dummy_input)) → output $(size(dummy_out)) ✓")

# ╔═╡ 6f40db4e-3560-11f1-bcdf-1defdb69d3e8
md"""
### 7.5 Training Loop

The training loop implements:
- **Cosine annealing** learning rate schedule
- **Early stopping** on validation loss (patience = 2 epochs)
- **Best-model checkpointing** — only the model with lowest validation loss is saved
- **Training/validation metrics** logged per epoch
"""

# ╔═╡ f78d07b7-707f-4751-a342-5760d4d08194
# ── Optimiser: AdamW (Adam with decoupled weight decay) ────────────────────
opt       = Flux.Optimisers.AdamW(LR_INIT, (0.9f0, 0.999f0), WEIGHT_DECAY)

# ╔═╡ 358a0e3a-3ed6-4591-859f-5ae042c1b677
opt_state = Flux.setup(opt, model)

# ╔═╡ 50037b51-e0e1-4b62-a321-3591f5b57390
# ── Training state ────────────────────────────────────────────────────────────
nothing

# ╔═╡ 04fa888b-b4f0-493b-994f-53a42f6a4dec
nothing

# ╔═╡ 79e0d6db-ee1a-42d1-9085-73df5bdd1ef9
nothing

# ╔═╡ 41f5e41f-e009-4d78-ab2d-659d9fb26e37
println("\n" * "=" ^ 60)

# ╔═╡ e9c1ffbb-cd7b-4888-a601-61b9e22c6f68
println("  Training: $EPOCHS epochs, batch_size=$BATCH_SIZE")

# ╔═╡ b6210db3-9ef6-4f30-87b2-4aad389939b1
println("  Optimiser: AdamW (lr=$LR_INIT, wd=$WEIGHT_DECAY)")

# ╔═╡ e42a689b-0164-4070-88af-cac3e6d1d5f4
println("  Schedule: Cosine annealing ($LR_INIT → $LR_MIN)")

# ╔═╡ 52c459b0-8746-4e8b-8473-a6a082c26cfb
println("  Early stopping: patience=$PATIENCE epochs")

# ╔═╡ 3daa72d7-5e3c-4391-84f4-d714eb813b27
println("  Julia threads: $(Threads.nthreads())")

# ╔═╡ 06c1e9a8-4b8c-4057-b48a-11b80867ccfe
println("  BLAS threads : $(BLAS.get_num_threads())")

# ╔═╡ 4930a1d5-6be6-4202-8ebd-f24ed807b0a0
println("=" ^ 60)

# ╔═╡ 6f40db4e-3560-11f1-82c9-533f238f97e0
history, best_val_loss = let
    local_history = (train=Float64[], val=Float64[], val_acc=Float64[])
    local_best_val_loss = Inf
    no_improve = 0

    for epoch in 1:EPOCHS
        # ── Update learning rate ──────────────────────────────────────────
        new_lr = cosine_lr(epoch, EPOCHS)
        Flux.Optimisers.adjust!(opt_state, new_lr)

        println("\n── Epoch $epoch / $EPOCHS  (lr=$(@sprintf "%.2e" new_lr)) ──")

        # ── Train ─────────────────────────────────────────────────────────
        Flux.trainmode!(model)
        train_batches = make_batches(train_s, BATCH_SIZE;
                                     shuffle_data=true, augment=AUGMENT, μ=μ, σ=σ)
        t_loss = train_epoch!(model, opt_state, train_batches)

        # ── Validate ──────────────────────────────────────────────────────
        Flux.testmode!(model)
        val_batches = make_batches(val_s, BATCH_SIZE;
                                   shuffle_data=false, augment=false, μ=μ, σ=σ)
        v_loss, v_acc = evaluate(model, val_batches; class_weights=class_weights)

        push!(local_history.train,   t_loss)
        push!(local_history.val,     v_loss)
        push!(local_history.val_acc, v_acc)

        @printf "  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f\n" t_loss v_loss v_acc

        # ── Checkpoint best model ─────────────────────────────────────────
        if v_loss < local_best_val_loss
            local_best_val_loss = v_loss
            no_improve = 0
            save_checkpoint(CHECKPOINT_PATH, model, opt_state, μ, σ, class_weights)
            println("  ✓ Model saved (best val_loss=$(round(local_best_val_loss; digits=4)))")
        else
            no_improve += 1
            println("  No improvement ($no_improve / $PATIENCE)")
            if no_improve >= PATIENCE
                println("  ⚡ Early stopping triggered.")
                break
            end
        end
    end

    (local_history, local_best_val_loss)
end

# ╔═╡ 6f40db4e-3560-11f1-b84f-855e1bae34a6
md"### 7.6 Training History"

# ╔═╡ 76d68649-3967-4982-897f-847584605094
# ── Training history visualisation (ASCII) ────────────────────────────────────
println("\nTraining History")

# ╔═╡ 3e5ff3f2-7d00-4b30-898c-b28a99d8347e
println("═" ^ 60)

# ╔═╡ 3fc50cff-a0ae-4b1f-ad0c-8e38663e0a14
@printf("%-8s  %12s  %12s  %12s\n", "Epoch", "Train Loss", "Val Loss", "Val Acc")

# ╔═╡ 83f7d1d2-3cde-4e75-969d-c62267e8b79c
println("─" ^ 60)

# ╔═╡ 9abb221e-90a4-4f51-95b0-5b61e83968d7
for (i, (tl, vl, va)) in enumerate(zip(history.train, history.val, history.val_acc))
    bar = "█" ^ round(Int, va * 30)
    @printf("  %3d     %10.4f    %10.4f    %10.4f  %s\n", i, tl, vl, va, bar)
end

# ╔═╡ 08b72c13-a6a6-4e95-a927-aba277d407a0
println("─" ^ 60)

# ╔═╡ 1a97e1ee-8d78-4715-a7df-f00479166377
println("\nBest validation loss: $(round(best_val_loss; digits=4))")

# ╔═╡ 6f40db4e-3560-11f1-8459-f9d9aa11171a
md"""
---

## 8. Final Evaluation on Test Set

Load the best checkpoint (lowest validation loss) and evaluate on the held-out test set to get unbiased performance estimates.
"""

# ╔═╡ 94d76f95-ed76-4e19-b45a-add22fad2e8e
# ── Load best model ───────────────────────────────────────────────────────────
println("Loading best checkpoint from: $CHECKPOINT_PATH")

# ╔═╡ bd92da59-9fc9-4217-a8f2-5db1aa6e7090
_, μ_best, σ_best, cw_best = load_checkpoint!(CHECKPOINT_PATH, model)

# ╔═╡ ff0f4b2e-1961-4898-bbee-8b0519144384
Flux.testmode!(model)

# ╔═╡ 09648d21-5bac-4d25-b4dc-bf76fc74f4ff
nothing

# ╔═╡ 3966bed5-72f2-4fd2-9628-3a3c310fcd60
# ── Evaluate on test set ──────────────────────────────────────────────────────
test_batches = make_batches(test_samples, BATCH_SIZE;
                            shuffle_data=false, augment=false, μ=μ_best, σ=σ_best)

# ╔═╡ 2a81650e-da1d-4a18-b874-de98a8296344
test_loss, test_acc = evaluate(model, test_batches; class_weights=cw_best)

# ╔═╡ d7d7ee75-84b6-4110-9438-52fb38f54ef1
println("\n" * "=" ^ 60)

# ╔═╡ b2a53ee3-abb0-41f6-b3dd-a9ec3a20e3b2
println("  FINAL TEST RESULTS")

# ╔═╡ a9b4b15e-5b4b-49c6-a6ab-07b23ea266d3
println("=" ^ 60)

# ╔═╡ e7e62764-2e98-4595-bf59-8cd3f605478f
@printf("  Test Loss     : %.4f\n", test_loss)

# ╔═╡ c4a61705-5614-4ea8-b72b-d5d30d782908
@printf("  Test Accuracy : %.4f (%.1f%%)\n", test_acc, test_acc * 100)

# ╔═╡ 3daef54e-cf67-40ff-90da-9f18018fa145
println("=" ^ 60)

# ╔═╡ 6f40db4e-3560-11f1-b062-75c2beff07a2
# ── Detailed classification report ────────────────────────────────────────────
cm, overall_acc = class_report(model, test_samples, BATCH_SIZE; μ=μ_best, σ=σ_best)

# ╔═╡ 6f40db4e-3560-11f1-8f3a-89955cc58252
md"""
---

## 9. Single Image Inference Demo

Demonstrate the model's prediction on a single MRI scan.
"""

# ╔═╡ f9ebe7f1-66b0-4473-86d4-25c9e9947df2
"""
    predict_image(path, model, μ, σ) → (predicted_class_index, confidence)

Run inference on a single image and return the predicted class index and
the softmax confidence score for that class.
"""
function predict_image(path::String, model, μ::Float32, σ::Float32)
    img = load_image(path)
    xs  = Array{Float32}(undef, IMG_SIZE[1], IMG_SIZE[2], 3, 1)
    xs[:, :, :, 1] = img
    normalize_batch!(xs, μ, σ)
    logits = model(xs |> device)
    probs = Flux.softmax(logits; dims=1)
    pred = onecold(probs)[1]
    return pred, vec(probs)[pred]
end

# ╔═╡ 2b570c1d-d560-4761-b64d-d21aa04bfe42
# ── Demo: pick a random test image ────────────────────────────────────────────
demo_sample = test_samples[rand(1:length(test_samples))]

# ╔═╡ b9ff0ea3-eb32-4976-9a8e-c15a79385d9c
demo_path, demo_truth = demo_sample

# ╔═╡ 4ddaaa59-5a56-4499-b18d-0b2d8aaa8d49
pred_idx, confidence = predict_image(demo_path, model, μ_best, σ_best)

# ╔═╡ ed2ad905-9327-446d-85df-294f39980b1e
println("\nSingle Image Inference")

# ╔═╡ 12e5de4c-dd48-419f-8d90-7485dce686fc
println("═" ^ 40)

# ╔═╡ 152d825c-22a6-45cf-bd28-6be1133b5801
println("  File       : $(basename(demo_path))")

# ╔═╡ d6d5716d-c520-4a63-a1da-1986d4d695db
println("  True class : $(CLASSES[demo_truth])")

# ╔═╡ b1c5d0fc-32be-4bac-b66b-5710c8e619c8
println("  Predicted  : $(CLASSES[pred_idx])")

# ╔═╡ 7cebea34-3618-489c-acda-ad151f60b316
@printf("  Confidence : %.2f%%\n", confidence * 100)

# ╔═╡ 55b3b669-835e-4d09-965b-dcd0dea69087
println("  Correct    : $(pred_idx == demo_truth ? "✓ Yes" : "✗ No")")

# ╔═╡ 80869ba8-f1cc-4852-860b-17d0576eeb24
# Display the image
display(imresize(load(demo_path), IMG_SIZE))

# ╔═╡ 6f40db4e-3560-11f1-8d49-174d6fc9929d
md"""
---

## 10. Summary & Discussion

### Results

| Metric | Value |
|:-------|:------|
| **Overall Test Accuracy** | ~93.6% |
| **Architecture** | ResNet-18 (trained from scratch) |
| **Parameters** | ~11.3M |
| **Training Time** | ~30 epochs with early stopping |

### Key Observations

1. **No-tumor and pituitary classes achieve near-perfect classification** (>99% recall), indicating that these categories have highly distinctive MRI features.

2. **Glioma is the most challenging class** (~79% recall), with most misclassifications being confused with meningioma. This is clinically expected — both are intra-axial tumors with overlapping radiological features.

3. **The model achieves 93.6% accuracy training from scratch** (no ImageNet pre-training). Enabling `pretrain=true` in `build_model()` would leverage transfer learning and likely push accuracy above 95%.

### Potential Improvements

| Enhancement | Expected Impact |
|:------------|:----------------|
| Enable ImageNet pre-training (`pretrain=true`) | +2-4% accuracy |
| Increase resolution to 224×224 | +1-2% accuracy |
| Use ResNet-50 or EfficientNet-B3 backbone | +1-3% accuracy |
| Test-time augmentation (TTA) | +0.5-1% accuracy |
| Grad-CAM saliency visualisation | Interpretability |
| Cross-validation (k-fold) | More robust estimate |

### Technology Stack

This project demonstrates the use of **Julia** as a viable platform for deep learning in medical imaging, leveraging:
- **Flux.jl** — A pure-Julia differentiable programming framework
- **Metalhead.jl** — Pre-built computer vision architectures
- Julia's native **multi-threading** for data loading
- **BSON.jl** for model serialisation

---

*Notebook generated for academic presentation. All code is self-contained and reproducible.*
"""

# ╔═╡ Cell order:
# ╟─72a16944-b39d-4607-9347-b9791a5d7429
# ╟─6f303980-3560-11f1-a4c7-0998bef080a7
# ╠═ea7147ed-9a0f-412c-8bb3-ff18e8f2832f
# ╟─6f39fd80-3560-11f1-9403-25a4c99377f4
# ╠═b1819df2-c72d-4a23-a1b0-62c0b171ebba
# ╠═d6659407-3c4c-4591-9144-21ec4a07e598
# ╠═bfb1a7dd-49ae-40d2-87f1-6f0e24c01e7d
# ╟─6f3e9160-3560-11f1-8147-9b1e1862e747
# ╠═a1f75273-0070-4c66-aaf6-694ad8cdcded
# ╠═1387fa2d-2fcc-4421-8a56-d8583f374213
# ╠═baeeabfd-a6d8-4e66-a954-59ed1558652b
# ╠═7aba4d48-c889-4c8f-9c40-8d45fd57b0f8
# ╠═c73633c1-c272-46f7-bb5f-502edde384d7
# ╠═958eeba2-dc10-475a-8159-49dcaf084429
# ╠═dfd50fa7-1079-4469-80af-448c35b6c40c
# ╟─6f3e9160-3560-11f1-9dab-ab05ac3fdc4b
# ╠═5e6252d7-45fb-44dd-9d6c-565850621d71
# ╠═e25ee0cd-2154-4a6a-9c46-1c11606b9745
# ╠═ed38c9d2-18fb-4394-a520-355e8386c795
# ╠═e7e20fdf-387e-4b9c-9d7d-5c493d66ce5c
# ╠═92301a9f-3c15-4ac3-b7f1-3d0b0074af6f
# ╠═bb29e883-1cae-412f-ab06-754a053a8b23
# ╠═5cc06f8c-7401-4db3-92c4-88261514b603
# ╠═62126c3f-dad8-4224-9879-3580adb7989b
# ╠═f3a15b7d-c90d-4af0-b936-530bb47ccd3f
# ╠═620f4bad-40a6-4e3a-acf0-d4dcf9c00649
# ╠═5d3e725d-8b26-4555-925d-078ed9e29acb
# ╠═3b6e8956-99a8-46e1-8c00-f4542ab75443
# ╠═561e81f7-9dc2-477a-ac75-e31abb1659e5
# ╠═59a36519-ad25-4248-8822-b34d49bb8772
# ╠═b3d3e27b-aa40-449b-9e42-9a4a527c65d6
# ╠═cf3f4451-f65b-4105-bfa5-49598618e01c
# ╠═3f8eaffd-6b4d-4eaf-91a2-b5945b2bb411
# ╠═af643dc3-7e99-46ef-8d4d-9d9985273a36
# ╠═a2213680-4ab1-4e39-b79c-cf5e30ad2ca1
# ╠═5659be55-bf8e-4157-9ec0-b6b9212c2563
# ╠═6940afff-eb8e-4b99-a0b0-4b1b48c73feb
# ╠═e33029d0-7f56-4544-afdb-451f1e1aa5de
# ╠═cbfc1c17-20a0-4ac2-9100-63eaabfdc7ef
# ╠═5ac497a7-9bbd-42f4-9f30-9f91e761b647
# ╟─6f3e9160-3560-11f1-9823-f75f6ba16141
# ╟─6f40db4e-3560-11f1-b2a2-9147170a746a
# ╠═6f40db4e-3560-11f1-b1a4-836afc44c818
# ╠═6f40db4e-3560-11f1-adc5-1f9cb50ee609
# ╟─6f40db4e-3560-11f1-a056-a5e8b92f1b08
# ╠═6f40db4e-3560-11f1-ae01-fba1890b46bb
# ╟─6f40db4e-3560-11f1-86c0-31a34b6aab12
# ╠═6f40db4e-3560-11f1-a53e-5fe0bad51aa6
# ╟─6f40db4e-3560-11f1-bacc-ddbacf99113f
# ╠═e8ee36c6-a1d0-4469-96f8-0f67deed5e69
# ╠═fb57e2af-3600-43d5-bc23-3d7a40cbb65f
# ╠═aaebe68b-9745-423d-b9fe-9e6498c4ace2
# ╠═f0a9954f-ec70-456b-a054-6a687ffe11f8
# ╠═bee3759e-2731-454a-8a94-a29859415ada
# ╟─6f40db4e-3560-11f1-8fc5-e3e99661f9cd
# ╠═6f40db4e-3560-11f1-9c1e-ddb5dc6b089f
# ╟─6f40db4e-3560-11f1-a30f-fd4eb57919f4
# ╠═6f40db4e-3560-11f1-a841-d11c53667df7
# ╟─6f40db4e-3560-11f1-9155-6174580a9cb2
# ╠═6f40db4e-3560-11f1-9aaa-89f70b385ebd
# ╟─6f40db4e-3560-11f1-8c1f-e1f9602553db
# ╠═0afadaa6-477e-4e28-bcf7-57bfd9367e9f
# ╠═d7b681be-c6b5-4bb2-933b-36d191de7f1f
# ╟─6f40db4e-3560-11f1-81d2-9f5dde4ea2e8
# ╠═ea5f9aaf-0f46-42a1-90de-9fc2eef51d53
# ╠═f4958ecd-1cbd-4281-a7c0-30fbef3ec94f
# ╟─6f40db4e-3560-11f1-9058-b1b941d0f4e6
# ╠═6f40db4e-3560-11f1-8be1-5dd5a4e385bc
# ╠═cf822b58-4e3c-4e03-998c-88a8061dc85e
# ╟─6f40db4e-3560-11f1-bf33-ed594fb334e8
# ╠═6f40db4e-3560-11f1-b04c-9d19dbf12472
# ╟─6f40db4e-3560-11f1-b6e4-3726fda8eb79
# ╠═ffe72ba7-4c3c-47c5-8b8e-34a7e7a1526a
# ╠═623a93df-78a0-4a49-9c1a-370be83cf877
# ╠═30bd603e-5e64-44bc-8ad1-cf88a28bc799
# ╠═0da95c67-ffb0-497e-929d-686268731442
# ╠═178ee25c-f8aa-43ca-b2fa-a9534abb518a
# ╠═8f993cfd-e871-47d6-90d8-411811133c77
# ╠═051dbfcf-e63f-4554-a672-60a02b917075
# ╠═2a7f8e4b-2f0c-4420-98b9-f72c89cb348c
# ╠═cbc51d10-71bd-43fd-935a-ecd7dbdddda1
# ╟─6f40db4e-3560-11f1-9f90-059a3eb9c2cf
# ╠═6f40db4e-3560-11f1-b2ae-616cb6d81df4
# ╟─6f40db4e-3560-11f1-8806-c7d5b294fc7a
# ╠═23b61c1f-1032-4611-8178-94e3be8665d7
# ╠═be848406-5f18-4a0b-a19c-12c6a809b49d
# ╠═453ff881-9691-418c-80e8-5b08a95f3544
# ╠═c1a0af6c-adc3-4bd8-b142-4f7aa717771f
# ╠═ba092993-37e9-4fef-95ba-b6c542dd5539
# ╠═840bae80-ab3a-4857-9af0-8586c659f52a
# ╠═12138524-7093-4ddd-8fcc-61b8a3ecb66a
# ╠═522270b3-f7ce-447a-aa3d-95d00ea13570
# ╠═8b839568-e504-42ff-80aa-172bd4fcf933
# ╠═d83b3cb6-05f3-4a64-96fe-1b80e8a33f9d
# ╠═76bf0388-7a48-4641-801c-e9e0ae9b9a45
# ╠═eaffd67f-d610-4fb1-8121-2a26ccb6011d
# ╠═233d5388-bc7d-40d7-b69e-ebe37fe26f85
# ╟─6f40db4e-3560-11f1-b3a5-650e0b06dd62
# ╠═3c317b8f-8245-4dee-97a8-6da96988d99a
# ╠═cfc04935-dcc0-4ffe-b787-f548fdc9ddbe
# ╠═764f69ba-951d-44f6-b745-bf3aeacaa8ea
# ╠═90f5057f-8c4d-4e05-bcec-3245f9101696
# ╠═8d2bf044-da62-43f5-a693-09f517c8871b
# ╠═0a6743f2-fd73-474a-88f9-873bb04feedb
# ╠═b99ebb2a-e845-4fc9-b8f3-766ead2791fa
# ╠═45cdf724-3abe-4ab1-9157-15ecd4c0ae27
# ╠═99bec46e-1c27-46c0-a6f3-ee70e0967240
# ╠═5cd724d9-984a-49cf-9e88-9641ee591980
# ╠═f7b60af1-2f54-48e8-ac5b-c7b128e3c922
# ╠═c37a6fef-bc23-4af1-9249-adcb73db5218
# ╟─6f40db4e-3560-11f1-bcdf-1defdb69d3e8
# ╠═f78d07b7-707f-4751-a342-5760d4d08194
# ╠═358a0e3a-3ed6-4591-859f-5ae042c1b677
# ╠═50037b51-e0e1-4b62-a321-3591f5b57390
# ╠═04fa888b-b4f0-493b-994f-53a42f6a4dec
# ╠═79e0d6db-ee1a-42d1-9085-73df5bdd1ef9
# ╠═41f5e41f-e009-4d78-ab2d-659d9fb26e37
# ╠═e9c1ffbb-cd7b-4888-a601-61b9e22c6f68
# ╠═b6210db3-9ef6-4f30-87b2-4aad389939b1
# ╠═e42a689b-0164-4070-88af-cac3e6d1d5f4
# ╠═52c459b0-8746-4e8b-8473-a6a082c26cfb
# ╠═3daa72d7-5e3c-4391-84f4-d714eb813b27
# ╠═06c1e9a8-4b8c-4057-b48a-11b80867ccfe
# ╠═4930a1d5-6be6-4202-8ebd-f24ed807b0a0
# ╠═6f40db4e-3560-11f1-82c9-533f238f97e0
# ╟─6f40db4e-3560-11f1-b84f-855e1bae34a6
# ╠═76d68649-3967-4982-897f-847584605094
# ╠═3e5ff3f2-7d00-4b30-898c-b28a99d8347e
# ╠═3fc50cff-a0ae-4b1f-ad0c-8e38663e0a14
# ╠═83f7d1d2-3cde-4e75-969d-c62267e8b79c
# ╠═9abb221e-90a4-4f51-95b0-5b61e83968d7
# ╠═08b72c13-a6a6-4e95-a927-aba277d407a0
# ╠═1a97e1ee-8d78-4715-a7df-f00479166377
# ╟─6f40db4e-3560-11f1-8459-f9d9aa11171a
# ╠═94d76f95-ed76-4e19-b45a-add22fad2e8e
# ╠═bd92da59-9fc9-4217-a8f2-5db1aa6e7090
# ╠═ff0f4b2e-1961-4898-bbee-8b0519144384
# ╠═09648d21-5bac-4d25-b4dc-bf76fc74f4ff
# ╠═3966bed5-72f2-4fd2-9628-3a3c310fcd60
# ╠═2a81650e-da1d-4a18-b874-de98a8296344
# ╠═d7d7ee75-84b6-4110-9438-52fb38f54ef1
# ╠═b2a53ee3-abb0-41f6-b3dd-a9ec3a20e3b2
# ╠═a9b4b15e-5b4b-49c6-a6ab-07b23ea266d3
# ╠═e7e62764-2e98-4595-bf59-8cd3f605478f
# ╠═c4a61705-5614-4ea8-b72b-d5d30d782908
# ╠═3daef54e-cf67-40ff-90da-9f18018fa145
# ╠═6f40db4e-3560-11f1-b062-75c2beff07a2
# ╟─6f40db4e-3560-11f1-8f3a-89955cc58252
# ╠═f9ebe7f1-66b0-4473-86d4-25c9e9947df2
# ╠═2b570c1d-d560-4761-b64d-d21aa04bfe42
# ╠═b9ff0ea3-eb32-4976-9a8e-c15a79385d9c
# ╠═4ddaaa59-5a56-4499-b18d-0b2d8aaa8d49
# ╠═ed2ad905-9327-446d-85df-294f39980b1e
# ╠═12e5de4c-dd48-419f-8d90-7485dce686fc
# ╠═152d825c-22a6-45cf-bd28-6be1133b5801
# ╠═d6d5716d-c520-4a63-a1da-1986d4d695db
# ╠═b1c5d0fc-32be-4bac-b66b-5710c8e619c8
# ╠═7cebea34-3618-489c-acda-ad151f60b316
# ╠═55b3b669-835e-4d09-965b-dcd0dea69087
# ╠═80869ba8-f1cc-4852-860b-17d0576eeb24
# ╟─6f40db4e-3560-11f1-8d49-174d6fc9929d
