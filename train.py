import os
import argparse
import random
import shutil
from typing import List, Tuple, Optional

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sentencepiece as spm

from model.utils.init_model import init_model
from model.utils.checkpoint import load_checkpoint


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


import sentencepiece as spm
from typing import List
import os

class TextTokenizer:
    """
    SentencePiece BPE tokenizer wrapper with CTC-friendly vocab
    Layout:
      - 0: <blank>
      - 1: <unk>
      - 2..N+1: sentencepiece pieces
    """

    def __init__(self, sp_model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)

    @staticmethod
    def train_from_corpus(texts: List[str], out_dir: str, vocab_size: int = 5000):
        os.makedirs(out_dir, exist_ok=True)
        corpus_path = os.path.join(out_dir, "corpus.txt")
        with open(corpus_path, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text.strip() + "\n")
        
        model_prefix = os.path.join(out_dir, "spm")
        spm.SentencePieceTrainer.Train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type='bpe',
            input_sentence_size=2000000,
            shuffle_input_sentence=True,
            bos_id=-1,
            eos_id=-1,
            unk_id=0
        )
        return model_prefix + ".model"

    def vocab_size(self) -> int:
        return self.sp.get_piece_size() + 2

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs with offset for CTC
        Returns: List of token IDs where 0=blank, 1=unk, 2+=actual tokens
        """
        ids = self.sp.encode_as_ids(text)
        return [i + 2 for i in ids]

    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs back to text
        Args: tokens with CTC offset (0=blank, 1=unk, 2+=actual)
        Returns: Decoded text string
        """
        filtered = []
        for t in tokens:
            if t == 0:  # Skip blank
                continue
            if t == 1:  # Handle unk
                filtered.append(0)  # SP unk id is 0
            elif t >= 2:
                filtered.append(t - 2)
        
        if not filtered:
            return ""
        
        return self.sp.decode_ids(filtered)

    def save_vocab_txt(self, output_path: str):
        """Save vocabulary in text format for debugging"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("<blank> 0\n")
            f.write("<unk> 1\n")
            for i in range(self.sp.get_piece_size()):
                piece = self.sp.id_to_piece(i)
                f.write(f'{piece} {i+2}\n')

    def get_vocab_dict(self) -> dict:
        """Get vocab as id->token dictionary"""
        vocab = {
            0: "<blank>",
            1: "<unk>"
        }
        for i in range(self.sp.get_piece_size()):
            vocab[i + 2] = self.sp.id_to_piece(i)
        return vocab


def maybe_speed_perturb(waveform: torch.Tensor, sample_rate: int, speeds=(0.9, 1.0, 1.1)) -> torch.Tensor:
    speed = random.choice(speeds)
    if abs(speed - 1.0) < 1e-6:
        return waveform
    # Use sox effects for natural speed perturb
    effects = [["speed", str(speed)], ["rate", str(sample_rate)]]
    wav, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects)
    return wav


def compute_fbank(waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    if waveform.dtype != torch.float32:
        waveform = waveform.to(torch.float32)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000
    feats = kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=sample_rate,
    )
    return feats


def spec_augment(feats: torch.Tensor, num_freq_mask: int = 2, freq_mask_param: int = 15,
                 num_time_mask: int = 2, time_mask_param: int = 50) -> torch.Tensor:
    x = feats.clone()
    # Frequency masking
    for _ in range(num_freq_mask):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(0, x.size(1) - f))
        x[:, f0:f0 + f] = 0
    # Time masking
    for _ in range(num_time_mask):
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, max(0, x.size(0) - t))
        x[t0:t0 + t, :] = 0
    return x


class AudioDataset(Dataset):
    def __init__(self, tsv_path: str, tokenizer: TextTokenizer, use_speed_perturb: bool = True,
                 return_texts: bool = False):
        df = pd.read_csv(tsv_path, sep="\t")
        assert "wav" in df.columns and "txt" in df.columns, "TSV must contain 'wav' and 'txt' columns"
        self.paths = df["wav"].tolist()
        self.texts = df["txt"].tolist()
        self.use_speed_perturb = use_speed_perturb
        self.tokenizer = tokenizer
        self.return_texts = return_texts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        text = str(self.texts[idx])
        waveform, sr = torchaudio.load(path)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if self.use_speed_perturb:
            waveform = maybe_speed_perturb(waveform, sr)
            
        feats = compute_fbank(waveform, sr)
        token_ids = self.tokenizer.encode(text)
        if self.return_texts:
            return feats, torch.tensor(token_ids, dtype=torch.long), text
        return feats, torch.tensor(token_ids, dtype=torch.long)


def collate_fn(batch, apply_specaug: bool = True) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for sample in batch:
        feats, tgt = sample[0], sample[1]
        if apply_specaug:
            feats = spec_augment(feats)
        xs.append(feats)
        ys.append(tgt)
    
    xs_lens = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    ys_cat = torch.cat([y for y in ys]) if len(ys) > 0 else torch.tensor([], dtype=torch.long)
    ys_lens = torch.tensor([len(y) for y in ys], dtype=torch.long)
    
    return xs, xs_lens, ys_cat, ys_lens


def build_or_load_tokenizer(train_tsv: str, out_dir: str, vocab_size: int) -> TextTokenizer:
    spm_path = os.path.join(out_dir, "spm.model")
    if os.path.exists(spm_path):
        return TextTokenizer(spm_path)
    texts = pd.read_csv(train_tsv, sep="\t")["txt"].astype(str).tolist()
    spm_path = TextTokenizer.train_from_corpus(texts, out_dir, vocab_size)
    return TextTokenizer(spm_path)


def build_config_yaml(output_dir: str, vocab_size: int, d_model: int, num_blocks: int,
                      attention_heads: int, linear_units: int, dropout_rate: float) -> str:
    config = {
        "cmvn_file": None,
        "is_json_cmvn": False,
        "input_dim": 80,
        "output_dim": vocab_size,
        "encoder_conf": {
            "output_size": d_model,
            "attention_heads": attention_heads,
            "linear_units": linear_units,
            "num_blocks": num_blocks,
            "dropout_rate": dropout_rate,
            "positional_dropout_rate": dropout_rate,
            "attention_dropout_rate": 0.0,
            "input_layer": "conv2d",
            "pos_enc_layer_type": "rel_pos",
            "normalize_before": True,
            "static_chunk_size": 0,
            "use_dynamic_chunk": False,
            "positionwise_conv_kernel_size": 1,
            "macaron_style": True,
            "selfattention_layer_type": "rel_selfattn",
            "activation_type": "swish",
            "use_cnn_module": True,
            "cnn_module_kernel": 15,
            "causal": False,
            "cnn_module_norm": "batch_norm",
            "use_limited_chunk": False,
            "limited_decoding_chunk_sizes": [],
            "limited_left_chunk_sizes": [],
            "use_dynamic_conv": False,
            "use_context_hint_chunk": False,
            "right_context_sizes": [],
            "right_context_probs": [],
            "freeze_subsampling_layer": False,
        }
    }
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return config_path


class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    """Noam scheduler with explicit peak learning rate at warmup_steps.

    lr(step) = scale * min(step^-0.5, step * warmup^-1.5)
    where scale is chosen so lr(warmup_steps) == peak_lr.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, peak_lr: float, last_epoch: int = -1):
        self.warmup_steps = max(1, warmup_steps)
        self.peak_lr = peak_lr
        # Determine scale so that lr at warmup equals peak_lr
        self.scale = peak_lr * (self.warmup_steps ** 0.5)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        factor = min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        lr = self.scale * factor
        return [lr for _ in self.base_lrs]


def save_vocab_and_config(tokenizer: TextTokenizer, output_dir: str, config_path: str):
    os.makedirs(output_dir, exist_ok=True)
    vocab_txt = os.path.join(output_dir, "vocab.txt")
    tokenizer.save_vocab_txt(vocab_txt)
    # Save SentencePiece model next to vocab for consistent inference
    spm_path = None
    try:
        spm_path = tokenizer.sp.model_file()
    except Exception:
        spm_path = None
    if spm_path and isinstance(spm_path, str) and os.path.exists(spm_path):
        shutil.copy2(spm_path, os.path.join(output_dir, "spm.model"))
    else:
        # Fallback: write serialized model bytes if available
        try:
            blob = tokenizer.sp.serialized_model_proto()
        except Exception:
            blob = None
        if blob:
            with open(os.path.join(output_dir, "spm.model"), "wb") as f:
                f.write(blob)
    return vocab_txt


def average_checkpoints(src_dir: str, out_path: str, last_n: int = 10):
    ckpts = sorted([p for p in os.listdir(src_dir) if p.endswith(".pt")])
    if len(ckpts) == 0:
        return
    ckpts = ckpts[-last_n:]
    avg_state = None
    for p in ckpts:
        state = torch.load(os.path.join(src_dir, p), map_location="cpu", weights_only=True)
        if avg_state is None:
            avg_state = {k: v.clone().to(torch.float32) for k, v in state.items()}
        else:
            for k in avg_state.keys():
                avg_state[k] += state[k].to(torch.float32)
    for k in avg_state.keys():
        avg_state[k] /= float(len(ckpts))
    torch.save(avg_state, out_path)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, scaler,
                    chunk_size: int, left_ctx: int, right_ctx: int,
                    grad_accum_steps: int = 1, max_grad_norm: float = 5.0) -> Tuple[float, int]:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    optimizer.zero_grad(set_to_none=True)
    for step, (xs, xs_lens, ys_cat, ys_lens) in enumerate(tqdm(dataloader, desc="train", leave=False)):
        xs = [x.to(device) for x in xs]
        ys_cat = ys_cat.to(device)
        ys_lens = ys_lens.to(device)

        # xs_origin_lens = torch.tensor(xs_lens, dtype=torch.int, device=device)
        xs_origin_lens = xs_lens.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            offset = torch.zeros(len(xs), dtype=torch.int, device=device)
            encoder_outs, encoder_lens, n_chunks, _, _, _ = model.encoder.forward_parallel_chunk(
                xs=xs,
                xs_origin_lens=xs_origin_lens,
                chunk_size=chunk_size,
                left_context_size=left_ctx,
                right_context_size=right_ctx,
                offset=offset,
            )
            # Rearrange to padded batch
            enc_padded, enc_masks = model.encoder.rearrange(encoder_outs, xs_origin_lens, n_chunks)
            input_lengths = enc_masks.squeeze(1).sum(dim=1).to(torch.int)
            log_probs = model.ctc.log_softmax(enc_padded)  # (B, T, V)
            log_probs = log_probs.transpose(0, 1)  # (T, B, V)
            # CTCLoss with targets concatenated
            loss = torch.nn.functional.ctc_loss(
                log_probs,
                ys_cat,
                input_lengths,
                ys_lens,
                blank=0,
                reduction="sum",
                zero_infinity=True,
            )
            # Normalize by tokens to be length-robust
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


@torch.no_grad()
def evaluate(model, dataloader, device, chunk_size: int, left_ctx: int, right_ctx: int) -> Tuple[float, int]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for xs, xs_lens, ys_cat, ys_lens in tqdm(dataloader, desc="valid", leave=False):
        xs = [x.to(device) for x in xs]
        ys_cat = ys_cat.to(device)
        ys_lens = ys_lens.to(device)
        # xs_origin_lens = torch.tensor(xs_lens, dtype=torch.int, device=device)
        xs_origin_lens = xs_lens.to(device)
        offset = torch.zeros(len(xs), dtype=torch.int, device=device)
        encoder_outs, encoder_lens, n_chunks, _, _, _ = model.encoder.forward_parallel_chunk(
            xs=xs,
            xs_origin_lens=xs_origin_lens,
            chunk_size=chunk_size,
            left_context_size=left_ctx,
            right_context_size=right_ctx,
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
        total_loss += (loss / norm).item() * norm.item()
        total_tokens += norm.item()
    avg_loss = total_loss / max(1, total_tokens)
    return avg_loss, total_tokens


def save_model_checkpoint(model, output_dir: str, step: int):
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"ckpt_step_{step}.pt"))


def main():
    parser = argparse.ArgumentParser(description="Train Chunkformer (CTC) - small GPU friendly")
    parser.add_argument("--train_tsv", type=str, required=True, help="Path to train tsv (wav, txt)")
    parser.add_argument("--valid_tsv", type=str, required=False, default=None, help="Path to valid tsv")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--peak_lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--vocab_size", type=int, default=5000)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--left_context", type=int, default=128)
    parser.add_argument("--right_context", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--linear_units", type=int, default=2048)
    parser.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every N optimizer steps")
    parser.add_argument("--avg_last_n", type=int, default=10, help="Average last N checkpoints at the end")
    parser.add_argument("--dropout", type=float, default=0.0, help="Encoder dropout (also positional)")
    parser.add_argument("--disable_specaug", action="store_true", help="Disable SpecAugment during training")
    parser.add_argument("--disable_speed_perturb", action="store_true", help="Disable speed perturbation during training")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # Build/load tokenizer and export vocab
    tokenizer = build_or_load_tokenizer(args.train_tsv, args.output_dir, args.vocab_size)
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

    # Init model
    with open(config_path, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    model = init_model(configs, config_path)

    device = torch.device(args.device)
    model = model.to(device)

    # Data
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
    valid_loader = None
    if args.valid_tsv and os.path.exists(args.valid_tsv):
        valid_ds = AudioDataset(args.valid_tsv, tokenizer, use_speed_perturb=False)
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_fn(b, apply_specaug=False),
            pin_memory=(device.type == "cuda"),
        )

    # Optim, sched, amp
    optimizer = torch.optim.Adam(model.parameters(), lr=args.peak_lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    scheduler = NoamLR(optimizer, warmup_steps=args.warmup_steps, peak_lr=args.peak_lr)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    global_step = 0
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_tokens = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            scaler,
            chunk_size=args.chunk_size,
            left_ctx=args.left_context,
            right_ctx=args.right_context,
            grad_accum_steps=args.accum_steps,
        )
        print(f"Train loss (CTC): {train_loss:.4f}")

        # Save periodic checkpoints by steps
        global_step += len(train_loader)
        if (epoch == args.epochs) or (args.save_every > 0 and global_step % args.save_every == 0):
            save_model_checkpoint(model, args.output_dir, global_step)

        if valid_loader is not None:
            val_loss, _ = evaluate(model, valid_loader, device, args.chunk_size, args.left_context, args.right_context)
            print(f"Val loss (CTC): {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))

    # Final save and averaging last-N checkpoints
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    if os.path.isdir(ckpt_dir) and len(os.listdir(ckpt_dir)) > 0:
        avg_path = os.path.join(args.output_dir, "pytorch_model.bin")
        average_checkpoints(ckpt_dir, avg_path, last_n=args.avg_last_n)
        print(f"Saved averaged checkpoint to {avg_path}")
    else:
        # Save last model if no checkpoints
        torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))


if __name__ == "__main__":
    main()


