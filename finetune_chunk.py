import os
import argparse
import yaml
import random
import torch
from torch.utils.data import DataLoader

from train import (
    TextTokenizer,
    AudioDataset,
    collate_fn,
    NoamLR,
    build_or_load_tokenizer,
    build_config_yaml,
    save_vocab_and_config,
    init_model,
)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_dynamic_context():
    # Dynamic limited-context policy: randomize chunk and contexts
    # Example ranges adapted from paper spirit; adjust as needed
    chunk_size = random.choice([32, 48])
    left_ctx = random.choice([64, 128])
    right_ctx = random.choice([32, 64])
    return chunk_size, left_ctx, right_ctx


def train_one_epoch_chunk(model, dataloader, optimizer, scheduler, device, scaler,
                          grad_accum_steps: int = 1, max_grad_norm: float = 5.0):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    optimizer.zero_grad(set_to_none=True)

    for step, (xs, xs_lens, ys_cat, ys_lens) in enumerate(dataloader):
        xs = [x.to(device) for x in xs]
        ys_cat = ys_cat.to(device)
        ys_lens = ys_lens.to(device)

        xs_origin_lens = xs_lens.to(device)
        c, l, r = sample_dynamic_context()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            offset = torch.zeros(len(xs), dtype=torch.int, device=device)
            encoder_outs, encoder_lens, n_chunks, _, _, _ = model.encoder.forward_parallel_chunk(
                xs=xs,
                xs_origin_lens=xs_origin_lens,
                chunk_size=c,
                left_context_size=l,
                right_context_size=r,
                offset=offset,
            )
            enc_padded, enc_masks = model.encoder.rearrange(encoder_outs, xs_origin_lens, n_chunks)
            input_lengths = enc_masks.squeeze(1).sum(dim=1).to(torch.int)
            log_probs = model.ctc.log_softmax(enc_padded).transpose(0, 1)

            loss = torch.nn.functional.ctc_loss(
                log_probs,
                ys_cat,
                input_lengths,
                ys_lens,
                blank=0,
                reduction="sum",
                zero_infinity=True,
            )
            norm = ys_lens.sum().clamp_min(1)
            loss = loss / norm

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.detach().item() * norm.item()
        total_tokens += norm.item()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    avg_loss = total_loss / max(1, total_tokens)
    return avg_loss, total_tokens


def main():
    parser = argparse.ArgumentParser(description="Chunk fine-tuning for Chunkformer (CTC)")
    parser.add_argument("--train_tsv", type=str, required=True)
    parser.add_argument("--valid_tsv", type=str, default=None)
    parser.add_argument("--init_model_dir", type=str, required=True, help="Directory of full-context checkpoint")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--peak_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--vocab_size", type=int, default=5000)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--linear_units", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--disable_specaug", action="store_true")
    parser.add_argument("--disable_speed_perturb", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # Use same tokenizer as full-context training
    init_model_dir = args.init_model_dir
    init_spm_path = os.path.join(init_model_dir, "spm.model")
    init_vocab_path = os.path.join(init_model_dir, "vocab.txt")
    tokenizer = TextTokenizer(init_spm_path)
    
    vocab_size = tokenizer.vocab_size()
    config_path = build_config_yaml(
        args.output_dir,
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_blocks=args.num_blocks,
        attention_heads=args.attention_heads,
        linear_units=args.linear_units,
        dropout_rate=args.dropout,
    )
    save_vocab_and_config(tokenizer, args.output_dir, config_path)

    with open(config_path, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    model = init_model(configs, config_path)

    # Load pre-trained full-context weights
    init_ckpt = os.path.join(args.init_model_dir, "pytorch_model.bin")
    state = torch.load(init_ckpt, map_location="cpu")
    _ = model.load_state_dict(state, strict=False)

    device = torch.device(args.device)
    model = model.to(device)

    train_ds = AudioDataset(
        args.train_tsv,
        tokenizer,
        use_speed_perturb=(not args.disable_speed_perturb),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, apply_specaug=(not args.disable_specaug)),
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.peak_lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    scheduler = NoamLR(optimizer, warmup_steps=args.warmup_steps, peak_lr=args.peak_lr)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, _ = train_one_epoch_chunk(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            scaler,
            grad_accum_steps=1,
        )
        print(f"Train loss (CTC): {train_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
    print("Saved chunk fine-tuned checkpoint.")


if __name__ == "__main__":
    main()


