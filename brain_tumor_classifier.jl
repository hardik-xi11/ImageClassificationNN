# Model parameters: 11,308,868

"""
── Final Evaluation on Test Set ─────────────────────────
  test_loss=1.0003  test_acc=0.6825

── Per-class Results ──────────────────────────────
Class           Precision    Recall        F1
──────────────────────────────────────────────────
glioma             0.686     0.475     0.561
meningioma         0.593     0.708     0.645
notumor            0.691     0.955     0.802
pituitary          0.809     0.593     0.684
──────────────────────────────────────────────────
Overall accuracy: 0.6825

Training history (val accuracy):
  Epoch  1 │ ██████████████████████ 0.7268
  Epoch  2 │ ████████████████████████ 0.8089
  Epoch  3 │ █████████████████████████ 0.8429
  Epoch  4 │ ██████████████████████████ 0.8589
  Epoch  5 │ ██████████████████████████ 0.8607
  Epoch  6 │ ██████████████████████████ 0.8661
  Epoch  7 │ █████████████████████████ 0.8464
  Epoch  8 │ ██████████████████████████ 0.8714
  Epoch  9 │ ██████████████████████████ 0.8750
  Epoch 10 │ ██████████████████████████ 0.8750
  Epoch 11 │ ██████████████████████████ 0.8679
  Epoch 12 │ ███████████████████████████ 0.8893
  Epoch 13 │ ██████████████████████████ 0.8786
  Epoch 14 │ ██████████████████████████ 0.8714
  Epoch 15 │ ██████████████████████████ 0.8661
  Epoch 16 │ ██████████████████████████ 0.8732
  Epoch 17 │ ███████████████████████████ 0.8857
  Epoch 18 │ ██████████████████████████ 0.8786
  Epoch 19 │ ██████████████████████████ 0.8821
  Epoch 20 │ ███████████████████████████ 0.8857
"""

"""
Brain Tumor MRI Classifier using Flux.jl
=========================================
Dataset structure expected:
  Training/
    glioma/       *.jpg / *.png
    meningioma/
    notumor/
    pituitary/
  Testing/
    glioma/
    meningioma/
    notumor/
    pituitary/

Usage:
  julia brain_tumor_classifier.jl
  julia brain_tumor_classifier.jl --train-dir Training --test-dir Testing --epochs 30

Performance optimisations used
-------------------------------
* Float32 throughout (halves memory, faster BLAS)
* Images resized to 128×128 (good resolution/speed trade-off)
* Parallel image loading with Threads.@threads
* DataLoader with pre-fetching (parallel=true)
* MobileNet-style depthwise-separable CNN - far fewer MACs than a plain CNN
* BatchNorm after every conv block - faster convergence, allows higher LR
* Cosine-annealing learning-rate schedule
* AdamW optimiser (Adam + weight decay) - regularisation without dropout overhead
* Label-smoothed cross-entropy loss - better generalisation
* Early stopping on validation loss
* BSON model checkpointing
* Optional CUDA / Metal GPU support
"""

# ─── Packages ────────────────────────────────────────────────────────────────
using Pkg
for pkg in ["Flux", "Images", "ImageIO", "FileIO", "ImageTransformations",
            "Statistics", "Random", "BSON", "ProgressMeter", "ArgParse", "Metalhead"]
    if !haskey(Pkg.project().dependencies, pkg)
        Pkg.add(pkg; io=devnull)
    end
end

using Flux, Metalhead, Images, ImageIO, FileIO, ImageTransformations, BSON, ProgressMeter, ArgParse, Printf
using Flux: onehotbatch, onecold, logitcrossentropy, state, loadmodel!
using Statistics: mean, std
using Random: shuffle!, randperm, seed!
using BSON: @save, @load

# ─── Optional GPU ─────────────────────────────────────────────────────────────
# Uncomment the backend that matches your hardware:
# using CUDA; device = gpu          # NVIDIA
# using Metal; device = gpu         # Apple Silicon
device = cpu                         # safe fallback

# ─── Constants ───────────────────────────────────────────────────────────────
const IMG_SIZE   = (128, 128)          # H × W fed to the network
const CLASSES    = ["glioma", "meningioma", "notumor", "pituitary"]
const N_CLASSES  = length(CLASSES)
const CLASS_IDX  = Dict(c => i for (i, c) in enumerate(CLASSES))

# ─── CLI Arguments ───────────────────────────────────────────────────────────
function parse_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--train-dir";  default = "Training";  help = "Path to Training directory"
        "--test-dir";   default = "Testing";   help = "Path to Testing directory"
        "--epochs";     default = 30;          arg_type = Int
        "--batch-size"; default = 32;          arg_type = Int
        "--lr";         default = 3f-4;        arg_type = Float32
        "--wd";         default = 1f-4;        arg_type = Float32   # weight decay
        "--seed";       default = 42;          arg_type = Int
        "--checkpoint"; default = "best_model.bson"
        "--val-split";  default = 0.2;         arg_type = Float64   # fraction of train → val
        "--no-augment"; action = :store_true;
                         help = "Disable training augmentation"
        "--inference"; action = :store_true;
                         help = "Load checkpoint and run inference instead of training"
        "--predict-image"; default = "";      help = "Path to a single image to classify"
    end
    return ArgParse.parse_args(s)
end

# ─── Data Loading ────────────────────────────────────────────────────────────

"""Return a Vector of (filepath, label_index) for every image under `root`."""
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

function augment_image(img::Array{Float32,3})
    if rand() < 0.5f0
        img = reverse(img, dims=1)
    end
    if rand() < 0.5f0
        img = reverse(img, dims=2)
    end

    # Add Random Rotations (up to ±15 degrees)
    if rand() < 0.5f0
        angle = (rand() - 0.5f0) * 0.5f0  # between -0.25 and 0.25 radians
        # Convert H×W×C back to Image object, rotate + resize back to original, then convert back
        c_img = colorview(RGB, permutedims(img, (3, 1, 2)))
        rot_img = ImageTransformations.imresize(ImageTransformations.imrotate(c_img, angle), (size(img, 1), size(img, 2)))
        img = Float32.(permutedims(channelview(rot_img), (2, 3, 1)))
    end

    # Add very small random Gaussian Noise
    if rand() < 0.3f0
        noise = randn(Float32, size(img)) .* 0.02f0
        img .= clamp.(img .+ noise, 0f0, 1f0)
    end

    if rand() < 0.4f0
        factor = 0.85f0 + 0.3f0 * rand(Float32)
        img .= clamp.(img .* factor, 0f0, 1f0)
    end
    if rand() < 0.4f0
        contrast = 0.8f0 + 0.4f0 * rand(Float32)
        img .= clamp.((img .- 0.5f0) .* contrast .+ 0.5f0, 0f0, 1f0)
    end
    return img
end

function compute_dataset_stats(samples::Vector{Tuple{String,Int}}; max_images::Int = 500)
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

function get_class_weights(samples::Vector{Tuple{String,Int}})
    counts = zeros(Int, N_CLASSES)
    for (_, lbl) in samples
        counts[lbl] += 1
    end
    max_count = maximum(counts)
    return Float32.(max_count ./ max.(counts, 1))
end

function normalize_batch!(xs::Array{Float32,4}, μ::Float32, σ::Float32)
    xs .= (xs .- μ) ./ σ
    return xs
end

function save_checkpoint(path::String, model, opt_state, μ::Float32, σ::Float32,
                         class_weights::Vector{Float32})
    model_state = Flux.state(model)
    @save path model_state opt_state μ σ class_weights
end

function load_checkpoint!(path::String, model)
    data = BSON.load(path)
    if haskey(data, :model_state)
        Flux.loadmodel!(model, data[:model_state])
    elseif haskey(data, :model)
        Flux.loadmodel!(model, data[:model])
    else
        error("Could not find :model_state or :model in checkpoint")
    end
    
    opt_state = get(data, :opt_state, nothing)
    μ = get(data, :μ, 0f0)
    σ = get(data, :σ, 1f0)
    class_weights = get(data, :class_weights, ones(Float32, N_CLASSES))
    
    return opt_state, μ, σ, class_weights
end

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

"""
Load and pre-process one image → Float32 H×W×3 array.
  1. Load → RGB
  2. Resize to IMG_SIZE (Lanczos3 for quality)
  3. Normalise to [0,1]  (ImageNet-style mean/std subtraction skipped for
     simplicity; per-dataset normalisation is computed on the fly in `build_batch`)
"""
function load_image(path::String)::Array{Float32, 3}
    img = load(path)
    img = imresize(img, IMG_SIZE)
    img = RGB{Float32}.(img)                        # ensure Float32 RGB
    arr = permutedims(channelview(img), (2, 3, 1))  # C×H×W → H×W×C
    return arr
end

"""Build a (H, W, C, N) Float32 tensor + one-hot (N_CLASSES, N) from a batch of samples."""
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

# ─── Model Architecture ──────────────────────────────────────────────────────
#
# MobileNet-style depthwise-separable CNN:
#   depthwise conv (groups = in_channels) + pointwise 1×1 conv
# Gives ~8-9× fewer MACs than a regular conv of the same size.
#
# Architecture (input: 128×128×3):
#  Stem: 3×3 Conv/2 → 64×64×32
#  Blocks:
#   dw_block(32→64,  stride=1) → 64×64×64
#   dw_block(64→128, stride=2) → 32×32×128
#   dw_block(128→128,stride=1) → 32×32×128
#   dw_block(128→256,stride=2) → 16×16×256
#   dw_block(256→256,stride=1) → 16×16×256
#   dw_block(256→512,stride=2) →  8×8×512
#  GlobalAvgPool → 512
#  Dense(512→N_CLASSES)

"""Depthwise-separable conv block with BatchNorm + ReLU."""
function dw_sep_block(in_ch::Int, out_ch::Int; stride::Int=1)
    return Chain(
        # Depthwise conv
        Conv((3, 3), in_ch => in_ch, relu;
             stride=stride, pad=1, groups=in_ch, bias=false),
        BatchNorm(in_ch),
        # Pointwise conv (1×1)
        Conv((1, 1), in_ch => out_ch, relu; bias=false),
        BatchNorm(out_ch),
    )
end

function build_model()
    # ── Transfer Learning with Metalhead.jl ─────────────────────────────
    # Load a pre-trained ResNet18 model. The feature extractor (backbone) 
    # is stored in the first layer block (`layers[1]`). 
    # Standard output features for ResNet18 is 512 dimensions.
    resnet = ResNet(18; pretrain=false)
    features = resnet.layers[1]

    return Chain(
        features,
        GlobalMeanPool(),      # Pool the spatial dimensions (H×W×512×N → 1×1×512×N)
        Flux.flatten,          # → 512×N
        Dense(512 => 256, relu),
        Dropout(0.3f0),
        Dense(256 => N_CLASSES)  # logits
    )
end

# ─── Loss (label smoothing) ──────────────────────────────────────────────────
"""
Label-smoothed cross-entropy.
  smoothed_target = (1-ε) * one_hot + ε / K
"""
function smooth_ce_loss(logits, labels; class_weights::Vector{Float32}=ones(Float32, N_CLASSES),
                         ε::Float32=0.1f0)
    K    = Float32(N_CLASSES)
    soft = (1f0 - ε) .* labels .+ ε ./ K   # smoothed targets
    logp = Flux.logsoftmax(logits; dims=1)
    losses = vec(-sum(soft .* logp; dims=1))
    truth = onecold(labels)
    sample_weights = class_weights[truth]
    return mean(sample_weights .* losses)
end

# ─── Cosine-Annealing LR Schedule ───────────────────────────────────────────
"""Cosine-annealed LR: decays from lr_max to lr_min over `T` epochs."""
cosine_lr(epoch::Int, T::Int; lr_max::Float32=3f-4, lr_min::Float32=1f-6) =
    lr_min + 0.5f0 * (lr_max - lr_min) * (1 + cos(Float32(π) * epoch / T))

# ─── Training Epoch ──────────────────────────────────────────────────────────
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

# ─── Evaluation ──────────────────────────────────────────────────────────────
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

# ─── DataLoader helper ───────────────────────────────────────────────────────
"""
Build pre-shuffled mini-batches from a list of (path, label) samples.
Images are loaded in parallel using Threads.@threads inside `build_batch`.
Returns a Vector of (xs, ys) tuples ready for training.
"""
function make_batches(samples::Vector{Tuple{String,Int}}, batch_size::Int;
                      shuffle_data::Bool=true, augment::Bool=false,
                      μ::Float32=0f0, σ::Float32=1f0)
    shuffle_data && shuffle!(samples)
    n       = length(samples)
    batches = Vector{Tuple{Array{Float32,4}, Matrix{Bool}}}()
    for start in 1:batch_size:n
        chunk = samples[start:min(start+batch_size-1, n)]
        push!(batches, build_batch(chunk; augment=augment, μ=μ, σ=σ))
    end
    return batches
end

# ─── Per-class accuracy report ───────────────────────────────────────────────
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
    # Overall accuracy
    acc = sum(cm[i,i] for i in 1:N_CLASSES) / sum(cm)
    @printf("Overall accuracy: %.4f\n\n", acc)

    println("── Confusion Matrix ───────────────────────────")
    print("      ")
    for i in 1:N_CLASSES
        @printf("%-6s ", first(CLASSES[i], 3))
    end
    println()
    for i in 1:N_CLASSES # true class (rows in typical display mapped from our predictions)
        # we want rows=truths, cols=preds to match standard conventions
        # cm[p, t] means cm[pred, truth], so cm[j, i] is what we want for row=truth(i), col=pred(j)
        @printf("%-5s ", first(CLASSES[i], 3))
        for j in 1:N_CLASSES
            @printf("%5d  ", cm[j, i])
        end
        println()
    end
    println()
end

# ─── Main ────────────────────────────────────────────────────────────────────

function main()
    args = parse_args()
    seed!(args["seed"])
    Flux.trainmode!(true)

    println("=" ^ 60)
    println("  Brain Tumor MRI Classifier — Flux.jl")
    println("=" ^ 60)
    println("  Train dir : ", args["train-dir"])
    println("  Test  dir : ", args["test-dir"])
    println("  Image size: $(IMG_SIZE[1])×$(IMG_SIZE[2])")
    println("  Epochs    : ", args["epochs"])
    println("  Batch size: ", args["batch-size"])
    println("  Threads   : $(Threads.nthreads())")
    println("  Device    : ", device == gpu ? "GPU" : "CPU")
    println("  Inference : ", args["inference"] ? "yes" : "no")
    println("  Predict image: ", args["predict-image"] == "" ? "none" : args["predict-image"])
    println("=" ^ 60)

    if args["inference"] || args["predict-image"] != ""
        model = build_model() |> device
        if !isfile(args["checkpoint"])
            error("Checkpoint not found: $(args["checkpoint"])\nPlease run training first or specify a valid checkpoint path.")
        end
        _, μ, σ, class_weights = load_checkpoint!(args["checkpoint"], model)
        Flux.testmode!(model)
        model = model |> device

        if args["predict-image"] != ""
            pred, score = predict_image(args["predict-image"], model, μ, σ)
            println("\nPrediction:")
            println("  file   : ", args["predict-image"])
            println("  class  : ", CLASSES[pred])
            println("  score  : ", round(score, digits=4))
        else
            println("\nNo image given, evaluating on entire test set...")
            test_samples = collect_samples(args["test-dir"])
            test_batches = make_batches(test_samples, args["batch-size"]; shuffle_data=false, augment=false, μ=μ, σ=σ)
            test_loss, test_acc = evaluate(model, test_batches; class_weights=class_weights)
            @printf "  test_loss=%.4f  test_acc=%.4f\n" test_loss test_acc
            class_report(model, test_samples, args["batch-size"]; μ=μ, σ=σ)
        end
        return
    end

    # ── Collect samples ──────────────────────────────────────────────────────
    train_samples = collect_samples(args["train-dir"])
    test_samples  = collect_samples(args["test-dir"])
    println("Train samples: $(length(train_samples))")
    println("Test  samples: $(length(test_samples))")

    # Class counts
    for cls in CLASSES
        n = count(s -> s[2] == CLASS_IDX[cls], train_samples)
        println("  $cls: $n")
    end

    # ── Validation split ─────────────────────────────────────────────────────
    train_s, val_s = stratified_split(train_samples, args["val-split"])
    println("\nTrain: $(length(train_s))  Val: $(length(val_s))  Test: $(length(test_samples))")

    μ, σ = compute_dataset_stats(train_s)
    class_weights = get_class_weights(train_s)
    println("  Train normalization: μ=$(round(μ, digits=4))  σ=$(round(σ, digits=4))")
    println("  Class weights: ", join(class_weights, ", "))

    # ── Build model ──────────────────────────────────────────────────────────
    model = build_model() |> device
    n_params = sum(length, Flux.params(model))
    @printf "\nModel parameters: %s\n" format_number(n_params)

    # ── Optimiser (AdamW = Adam + weight-decay) ───────────────────────────────
    lr_init   = args["lr"]
    opt       = Flux.Optimisers.AdamW(lr_init, (0.9f0, 0.999f0), args["wd"])
    opt_state = Flux.setup(opt, model)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = Inf
    patience      = 7           # early-stopping patience (epochs)
    no_improve    = 0
    history       = (train=Float64[], val=Float64[], val_acc=Float64[])

    for epoch in 1:args["epochs"]
        # Update LR (cosine annealing)
        new_lr = cosine_lr(epoch, args["epochs"]; lr_max=lr_init)
        Flux.Optimisers.adjust!(opt_state, new_lr)

        println("\n── Epoch $epoch / $(args["epochs"])  (lr=$(@sprintf "%.2e" new_lr)) ──")

        # ── Train ────────────────────────────────────────────────────────────
        Flux.trainmode!(model)
        augment = !get(args, "no_augment", false)
        train_batches = make_batches(train_s, args["batch-size"]; shuffle_data=true,
                                     augment=augment, μ=μ, σ=σ)
        t_loss = train_epoch!(model, opt_state, train_batches)

        # ── Validate ─────────────────────────────────────────────────────────
        Flux.testmode!(model)
        val_batches = make_batches(val_s, args["batch-size"]; shuffle_data=false,
                                   augment=false, μ=μ, σ=σ)
        v_loss, v_acc = evaluate(model, val_batches; class_weights=class_weights)

        push!(history.train,   t_loss)
        push!(history.val,     v_loss)
        push!(history.val_acc, v_acc)

        @printf "  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f\n" t_loss v_loss v_acc

        # ── Checkpoint ───────────────────────────────────────────────────────
        if v_loss < best_val_loss
            best_val_loss = v_loss
            no_improve    = 0
            save_checkpoint(args["checkpoint"], model, opt_state, μ, σ, class_weights)
            println("  ✓ Model saved (best val_loss=$(round(best_val_loss; digits=4)))")
        else
            no_improve += 1
            println("  No improvement ($no_improve / $patience)")
            no_improve >= patience && (println("  Early stopping."); break)
        end
    end

    # ── Load best model and evaluate on test set ─────────────────────────────
    println("\n── Final Evaluation on Test Set ─────────────────────────")
    _, μ, σ, class_weights = load_checkpoint!(args["checkpoint"], model)
    Flux.testmode!(model)
    model = model |> device

    test_batches = make_batches(test_samples, args["batch-size"]; shuffle_data=false,
                                 augment=false, μ=μ, σ=σ)
    test_loss, test_acc = evaluate(model, test_batches; class_weights=class_weights)
    @printf "  test_loss=%.4f  test_acc=%.4f\n" test_loss test_acc

    class_report(model, test_samples, args["batch-size"]; μ=μ, σ=σ)

    println("Training history (val accuracy):")
    for (i, a) in enumerate(history.val_acc)
        bar = "█" ^ round(Int, a * 30)
        @printf "  Epoch %2d │ %s %.4f\n" i bar a
    end
end

# ─── Utility ─────────────────────────────────────────────────────────────────
function format_number(n::Int)
    s = string(n)
    out = ""
    for (i, c) in enumerate(reverse(s))
        i > 1 && i % 3 == 1 && (out = "," * out)
        out = string(c) * out
    end
    return out
end

main()
