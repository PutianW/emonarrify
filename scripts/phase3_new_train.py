"""Phase 3 (new) training: emotion-only VITS finetune from VCTK pretrained.

Implements the spec from /home/ubuntu/.claude/plans/andrew-floofy-pancake.md:
  L_total^(G) = L_VITS_upstream + lambda_cls * L_cls + lambda_ortho * L_ortho

  L_cls   = CrossEntropy(emotion_classifier_head(emb_e[eid]), eid)  (lambda 1.0)
  L_ortho = mean over off-diagonal pairs of relu(cos_sim - 0.3)^2   (lambda 0.05)

Single-GPU, no DDP. Mirrors vits/train_ms.py train_and_evaluate() shape but:
  * uses TextAudioEmotionLoader (audio|eid|text)
  * passes eid= to net_g forward
  * adds two auxiliary loss terms with detailed grad-norm logging
  * step-based loop (config train.max_steps_total) instead of epoch-based

Init from weights/vits_vctk/pretrained_vctk.pth via utils.load_checkpoint
selective load: emb_g.weight (109,256) silently skipped, emb_e.weight (5,256)
keeps orthogonal init, emotion_classifier_head.* keeps PyTorch defaults,
enc_p / enc_q / flow / dp / dec receive pretrained weights.

Discriminator starts from random (no pretrained D). Burns in over first ~200
steps; OK because mel/kl/dur losses dominate early reconstruction.

Smoke test:
  PYTHONPATH=. python scripts/phase3_new_train.py \
      -c vits/configs/phase3_new_v1.json --max-steps 100

Production:
  PYTHONPATH=. python scripts/phase3_new_train.py \
      -c vits/configs/phase3_new_v1.json
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
for _name in ("numba", "numba.core", "numba.core.byteflow", "numba.core.interpreter",
              "librosa", "matplotlib", "phonemizer"):
    logging.getLogger(_name).setLevel(logging.WARNING)
logging.getLogger("phonemizer").setLevel(logging.ERROR)

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VITS_DIR = os.path.join(ROOT, "vits")
sys.path.insert(0, VITS_DIR)

import commons  # noqa: E402
import utils  # noqa: E402
from data_utils import TextAudioEmotionLoader, TextAudioEmotionCollate  # noqa: E402
from models import SynthesizerTrn, MultiPeriodDiscriminator  # noqa: E402
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss  # noqa: E402
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch  # noqa: E402
from text.symbols import symbols  # noqa: E402


def compute_ortho_penalty(emb_weight: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """Hinge-squared cosine-similarity penalty on emotion embedding rows.

    L = (2 / K(K-1)) * sum_{i<j} relu(cos(e_i, e_j) - threshold)^2
    """
    K = emb_weight.shape[0]
    if K < 2:
        return emb_weight.new_zeros(())
    normed = emb_weight / (emb_weight.norm(dim=1, keepdim=True) + 1e-8)
    sim = normed @ normed.T  # (K, K)
    eye = torch.eye(K, device=sim.device, dtype=sim.dtype)
    hinge = F.relu(sim - threshold) ** 2
    upper = torch.triu(hinge * (1.0 - eye), diagonal=1).sum()
    n_pairs = K * (K - 1) / 2
    return upper / n_pairs


def emb_cos_sim_metrics(emb_weight: torch.Tensor) -> tuple[float, float, float]:
    """Mean/max off-diagonal pairwise cosine sim + Frobenius norm."""
    with torch.no_grad():
        normed = emb_weight / emb_weight.norm(dim=1, keepdim=True)
        sim = normed @ normed.T
        K = emb_weight.shape[0]
        upper = torch.triu(sim - torch.eye(K, device=sim.device, dtype=sim.dtype), diagonal=1)
        n_pairs = K * (K - 1) // 2
        return upper.sum().item() / n_pairs, upper.max().item(), emb_weight.norm().item()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override config train.max_steps_total (smoke uses 100)")
    p.add_argument("--init-from", type=str, default="weights/vits_vctk/pretrained_vctk.pth",
                   help="Pretrained generator checkpoint (selective load)")
    p.add_argument("--no-init", action="store_true",
                   help="Skip pretrained init (random scratch run)")
    return p.parse_args()


def evaluate(net_g, eval_loader, hps, device, writer, global_step) -> None:
    net_g.eval()
    val_mels, val_clss = [], []
    correct = total = 0
    with torch.no_grad():
        for batch in eval_loader:
            x, x_lens, spec, spec_lens, y, y_lens, eids = [t.to(device) for t in batch]
            y_hat, _, _, ids_slice, _, _, _ = net_g(x, x_lens, spec, spec_lens, eid=eids)
            mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels,
                                    hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1), hps.data.filter_length,
                                              hps.data.n_mel_channels, hps.data.sampling_rate,
                                              hps.data.hop_length, hps.data.win_length,
                                              hps.data.mel_fmin, hps.data.mel_fmax)
            val_mels.append(F.l1_loss(y_mel, y_hat_mel).item() * hps.train.c_mel)
            cls_logits = net_g.emotion_classifier_head(net_g.emb_e(eids))
            val_clss.append(F.cross_entropy(cls_logits, eids).item())
            correct += (cls_logits.argmax(dim=-1) == eids).sum().item()
            total += eids.shape[0]
    val_mel = sum(val_mels) / max(len(val_mels), 1)
    val_cls = sum(val_clss) / max(len(val_clss), 1)
    val_acc = correct / max(total, 1)
    print(f"  [eval @ step {global_step}] val_mel={val_mel:.2f} val_cls={val_cls:.3f} val_cls_acc={val_acc:.3f}")
    writer.add_scalar("eval/loss/mel", val_mel, global_step)
    writer.add_scalar("eval/loss/cls", val_cls, global_step)
    writer.add_scalar("eval/classifier_acc", val_acc, global_step)


def main() -> None:
    args = parse_args()
    hps = utils.get_hparams_from_file(args.config)
    if args.max_steps is not None:
        hps.train.max_steps_total = args.max_steps

    torch.manual_seed(hps.train.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Phase 3 training requires CUDA"

    model_dir = hps.train.model_dir
    os.makedirs(model_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=model_dir)

    train_dataset = TextAudioEmotionLoader(hps.data.training_files, hps.data)
    eval_dataset = TextAudioEmotionLoader(hps.data.validation_files, hps.data)
    collate = TextAudioEmotionCollate()
    train_loader = DataLoader(train_dataset, batch_size=hps.train.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=hps.train.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True, collate_fn=collate)
    print(f"[init] train dataset = {len(train_dataset)} utts, eval = {len(eval_dataset)} utts")
    print(f"[init] batches/epoch = {len(train_loader)}, batch_size = {hps.train.batch_size}")

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        n_emotions=hps.data.n_emotions,
        **hps.model,
    ).to(device)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)

    optim_g = optim.AdamW(net_g.parameters(), hps.train.learning_rate,
                          betas=hps.train.betas, eps=hps.train.eps)
    optim_d = optim.AdamW(net_d.parameters(), hps.train.learning_rate,
                          betas=hps.train.betas, eps=hps.train.eps)

    if not args.no_init and args.init_from and os.path.isfile(args.init_from):
        utils.load_checkpoint(args.init_from, net_g, None)
        print(f"[init] Loaded VCTK pretrained generator: {args.init_from}")
    else:
        print(f"[init] Skipping pretrained init (no_init={args.no_init}, "
              f"file_exists={os.path.isfile(args.init_from) if args.init_from else False})")

    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay)
    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay)
    scaler = GradScaler(enabled=hps.train.fp16_run)

    max_steps = hps.train.max_steps_total
    log_int = hps.train.log_interval
    eval_int = hps.train.eval_interval
    save_int = hps.train.save_interval

    global_step = 0
    epoch = 1
    initial_loss_total = None
    initial_loss_cls = None
    final_loss_total = None
    final_loss_cls = None
    final_emb_e_grad = None
    final_dec_grad = None
    final_cls_head_grad = None
    final_grad_ratio = None
    t_start = time.time()

    print(f"[init] Starting training: max_steps={max_steps}, fp16={hps.train.fp16_run}")
    print(f"[init] lambda_cls={hps.train.lambda_cls} lambda_ortho={hps.train.lambda_ortho} "
          f"ortho_threshold={hps.train.ortho_threshold}")

    while global_step < max_steps:
        net_g.train()
        net_d.train()
        for batch in train_loader:
            if global_step >= max_steps:
                break
            x, x_lens, spec, spec_lens, y, y_lens, eids = [t.to(device, non_blocking=True) for t in batch]

            with autocast(enabled=hps.train.fp16_run):
                y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
                    (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lens, spec, spec_lens, eid=eids)

                mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels,
                                        hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
                y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
                y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1), hps.data.filter_length,
                                                  hps.data.n_mel_channels, hps.data.sampling_rate,
                                                  hps.data.hop_length, hps.data.win_length,
                                                  hps.data.mel_fmin, hps.data.mel_fmax)
                y_seg = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)

                y_d_hat_r, y_d_hat_g, _, _ = net_d(y_seg, y_hat.detach())
                with autocast(enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                    loss_disc_all = loss_disc
            optim_d.zero_grad(set_to_none=True)
            scaler.scale(loss_disc_all).backward()
            scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)

            with autocast(enabled=hps.train.fp16_run):
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y_seg, y_hat)
                with autocast(enabled=False):
                    loss_dur = torch.sum(l_length.float())
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                    loss_kl_v = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    g_for_cls = net_g.emb_e(eids)
                    cls_logits = net_g.emotion_classifier_head(g_for_cls)
                    loss_cls = F.cross_entropy(cls_logits, eids)
                    loss_ortho = compute_ortho_penalty(net_g.emb_e.weight, threshold=hps.train.ortho_threshold)
                    loss_gen_all = (loss_gen + loss_fm + loss_mel + loss_dur + loss_kl_v
                                    + hps.train.lambda_cls * loss_cls
                                    + hps.train.lambda_ortho * loss_ortho)
            optim_g.zero_grad(set_to_none=True)
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            emb_e_grad = (net_g.emb_e.weight.grad.norm().item()
                          if net_g.emb_e.weight.grad is not None else 0.0)
            cls_head_grad = (net_g.emotion_classifier_head[0].weight.grad.norm().item()
                             if net_g.emotion_classifier_head[0].weight.grad is not None else 0.0)
            dec_grad = (net_g.dec.conv_pre.weight.grad.norm().item()
                        if net_g.dec.conv_pre.weight.grad is not None else 0.0)
            grad_ratio = emb_e_grad / max(dec_grad, 1e-12)
            scaler.step(optim_g)
            scaler.update()

            if initial_loss_total is None:
                initial_loss_total = loss_gen_all.item()
                initial_loss_cls = loss_cls.item()
            final_loss_total = loss_gen_all.item()
            final_loss_cls = loss_cls.item()
            final_emb_e_grad = emb_e_grad
            final_dec_grad = dec_grad
            final_cls_head_grad = cls_head_grad
            final_grad_ratio = grad_ratio

            if global_step % log_int == 0:
                lr = optim_g.param_groups[0]["lr"]
                cls_acc = (cls_logits.argmax(dim=-1) == eids).float().mean().item()
                # top1-top2 logit margin keeps diagnostic value after train acc saturates at 1.0;
                # rising margin = classifier more confident = emb_e becoming more distinctive
                top2 = torch.topk(cls_logits.detach(), 2, dim=-1).values
                cls_margin = (top2[:, 0] - top2[:, 1]).mean().item()
                mean_off, max_off, emb_e_norm = emb_cos_sim_metrics(net_g.emb_e.weight)
                elapsed = time.time() - t_start
                step_per_s = (global_step + 1) / max(elapsed, 1e-3)
                print(f"[step {global_step:>5}] mel={loss_mel.item():.2f} kl={loss_kl_v.item():.2f} "
                      f"dur={loss_dur.item():.2f} fm={loss_fm.item():.2f} gen={loss_gen.item():.2f} "
                      f"cls={loss_cls.item():.3f}(acc={cls_acc:.2f}) ortho={loss_ortho.item():.4f} "
                      f"| total={loss_gen_all.item():.2f} disc={loss_disc_all.item():.2f} "
                      f"| g_grad[emb_e={emb_e_grad:.3f} cls={cls_head_grad:.3f} dec={dec_grad:.3f} "
                      f"ratio={grad_ratio:.3f}] "
                      f"| emb_e[cos_mean={mean_off:.4f} cos_max={max_off:.4f} norm={emb_e_norm:.2f}] "
                      f"| {step_per_s:.2f} step/s")
                sd = {
                    "loss/g/total": loss_gen_all.item(), "loss/g/mel": loss_mel.item(),
                    "loss/g/kl": loss_kl_v.item(), "loss/g/dur": loss_dur.item(),
                    "loss/g/fm": loss_fm.item(), "loss/g/gen": loss_gen.item(),
                    "loss/g/cls": loss_cls.item(), "loss/g/ortho": loss_ortho.item(),
                    "loss/d/total": loss_disc_all.item(), "learning_rate": lr,
                    "grad_norm/emb_e": emb_e_grad, "grad_norm/cls_head": cls_head_grad,
                    "grad_norm/dec": dec_grad, "grad_norm/emb_e_dec_ratio": grad_ratio,
                    "emb_e/cos_sim_mean_off_diag": mean_off,
                    "emb_e/cos_sim_max_off_diag": max_off,
                    "emb_e/frobenius_norm": emb_e_norm,
                    "classifier/train_acc": cls_acc,
                    "classifier/logit_margin": cls_margin,
                }
                for k, v in sd.items():
                    writer.add_scalar(k, v, global_step)

            if global_step > 0 and global_step % eval_int == 0:
                evaluate(net_g, eval_loader, hps, device, writer, global_step)
                net_g.train()
                net_d.train()

            if global_step > 0 and global_step % save_int == 0:
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                      os.path.join(model_dir, f"G_{global_step}.pth"))
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                      os.path.join(model_dir, f"D_{global_step}.pth"))

            global_step += 1

        scheduler_g.step()
        scheduler_d.step()
        epoch += 1

    elapsed = time.time() - t_start
    final_step_per_s = max_steps / max(elapsed, 1e-3)
    eta_20k = 20000 / final_step_per_s
    final_mean_off, final_max_off, final_norm = emb_cos_sim_metrics(net_g.emb_e.weight)

    print()
    print("=" * 78)
    print(f"Run summary: {max_steps} steps in {elapsed:.1f}s "
          f"({final_step_per_s:.2f} step/s, ETA 20k = {eta_20k/60:.1f} min = {eta_20k/3600:.2f} h)")
    print(f"  loss/g/total: init={initial_loss_total:.2f} -> final={final_loss_total:.2f}")
    print(f"  loss/g/cls:   init={initial_loss_cls:.3f} -> final={final_loss_cls:.3f}  "
          f"(random baseline ln 5 = {math.log(5):.3f})")
    print(f"  emb_e cos_sim mean off-diag: {final_mean_off:.4f} (init was 0.0)")
    print(f"  emb_e cos_sim max off-diag:  {final_max_off:.4f}")
    print(f"  emb_e frobenius norm:        {final_norm:.2f}")
    print(f"  Final grad norms: emb_e={final_emb_e_grad:.3f} cls={final_cls_head_grad:.3f} "
          f"dec={final_dec_grad:.3f} ratio={final_grad_ratio:.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
