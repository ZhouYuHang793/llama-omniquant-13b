#!/usr/bin/env python3
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path("/root/autodl-tmp/omniquant_final/code")
OMNI = ROOT / "quantize" / "omniquant.py"


def main() -> None:
    if not OMNI.is_file():
        raise FileNotFoundError(f"Missing source file: {OMNI}")

    text = OMNI.read_text(encoding="utf-8")

    if "[ROBUST] FP32 fallback" in text:
        print("[skip] robust v2 patch is already present")
        return

    start_anchor = '''            else:
                trainable_parameters = list(get_omni_parameters(qlayer, use_shift))
'''
    end_anchor = '''                del optimizer

            clear_temp_variable(qlayer)
'''

    start = text.find(start_anchor)
    end = text.find(end_anchor, start)

    if start < 0 or end < 0:
        raise RuntimeError(
            "Could not locate the robust-mode block. "
            "Make sure the v1 robust patch is committed and the source is unchanged."
        )

    replacement = r'''            else:
                trainable_parameters = list(get_omni_parameters(qlayer, use_shift))
                if not trainable_parameters:
                    raise RuntimeError("robust_mode found no learnable LET/LWC parameters")

                current_let_lr = float(args.let_lr)
                current_lwc_lr = float(args.lwc_lr)
                optimizer = _build_omni_optimizer(
                    qlayer, use_shift, current_let_lr, current_lwc_lr, args.wd
                )
                amp_enabled = not args.deactive_amp
                grad_scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

                for epochs in range(args.epochs):
                    retry = 0
                    epoch_skipped = False

                    while True:
                        # Retry hierarchy:
                        #   retry 0: original AMP path
                        #   retry 1+: full-precision forward/backward
                        #   retry 2+: sanitize any remaining non-finite gradient entries
                        fp32_fallback = retry >= 1
                        sanitize_gradients = retry >= 2

                        epoch_snapshot = _snapshot_parameters(trainable_parameters)
                        loss_list = []
                        norm_list = []
                        failure_reason = None

                        for j in range(args.nsamples // args.batch_size):
                            index = j * args.batch_size
                            optimizer.zero_grad(set_to_none=True)

                            if fp32_fallback:
                                batch_inps = quant_inps[
                                    index:index + args.batch_size,
                                ].float()
                                batch_targets = fp_inps[
                                    index:index + args.batch_size,
                                ].float()

                                batch_attention_mask = attention_mask_batch
                                if (
                                    isinstance(batch_attention_mask, torch.Tensor)
                                    and batch_attention_mask.is_floating_point()
                                ):
                                    batch_attention_mask = batch_attention_mask.float()

                                with nullcontext():
                                    smooth_and_quant_temporary(qlayer, args, is_llama)
                                    quant_out = qlayer(
                                        batch_inps,
                                        attention_mask=batch_attention_mask,
                                        position_ids=position_ids,
                                    )[0]
                                    loss = loss_func(
                                        batch_targets,
                                        quant_out.float(),
                                    )
                                    if args.aug_loss:
                                        batch_targets_2 = fp_inps_2[
                                            index:index + args.batch_size,
                                        ].float()
                                        loss += loss_func(
                                            batch_targets_2,
                                            quant_out.float(),
                                        )
                            else:
                                with traincast():
                                    smooth_and_quant_temporary(qlayer, args, is_llama)
                                    quant_out = qlayer(
                                        quant_inps[
                                            index:index + args.batch_size,
                                        ],
                                        attention_mask=attention_mask_batch,
                                        position_ids=position_ids,
                                    )[0]
                                    loss = loss_func(
                                        fp_inps[
                                            index:index + args.batch_size,
                                        ],
                                        quant_out,
                                    )
                                    if args.aug_loss:
                                        loss += loss_func(
                                            fp_inps_2[
                                                index:index + args.batch_size,
                                            ],
                                            quant_out,
                                        )

                            loss_value, loss_is_finite = _finite_float(loss)
                            if not loss_is_finite:
                                failure_reason = (
                                    f"non-finite loss={loss_value} at batch={j}"
                                )
                                break

                            if fp32_fallback:
                                loss.backward()
                            else:
                                grad_scaler.scale(loss).backward()
                                grad_scaler.unscale_(optimizer)

                            nonfinite_entries = 0
                            if sanitize_gradients:
                                for parameter in trainable_parameters:
                                    if parameter.grad is None:
                                        continue
                                    bad_mask = ~torch.isfinite(parameter.grad)
                                    bad_count = int(bad_mask.sum().item())
                                    if bad_count:
                                        nonfinite_entries += bad_count
                                        parameter.grad.nan_to_num_(
                                            nan=0.0,
                                            posinf=0.0,
                                            neginf=0.0,
                                        )

                                if nonfinite_entries:
                                    logger.warning(
                                        f"[ROBUST] sanitized {nonfinite_entries} "
                                        f"non-finite gradient entries at "
                                        f"layer={i}, epoch={epochs}, batch={j}"
                                    )

                            max_norm = (
                                float(args.grad_clip)
                                if args.grad_clip > 0
                                else float("inf")
                            )
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                trainable_parameters,
                                max_norm=max_norm,
                            )
                            norm_value, norm_is_finite = _finite_float(grad_norm)

                            if not norm_is_finite:
                                failure_reason = (
                                    f"non-finite grad_norm={norm_value} "
                                    f"at batch={j}"
                                )
                                optimizer.zero_grad(set_to_none=True)
                                break

                            if fp32_fallback:
                                optimizer.step()
                            else:
                                grad_scaler.step(optimizer)
                                grad_scaler.update()

                            loss_list.append(loss.detach().float().cpu())
                            norm_list.append(
                                torch.tensor(norm_value, dtype=torch.float32)
                            )

                        if failure_reason is None:
                            loss_mean = torch.stack(loss_list).mean()
                            norm_mean = torch.stack(norm_list).mean()
                            mode = "fp32" if fp32_fallback else "amp"
                            logger.info(
                                f"[ROBUST] layer {i} iter {epochs} "
                                f"loss:{loss_mean} norm:{norm_mean} "
                                f"retry:{retry} mode:{mode} "
                                f"let_lr:{current_let_lr:.3e} "
                                f"lwc_lr:{current_lwc_lr:.3e} "
                                f"max memory_allocated "
                                f"{torch.cuda.max_memory_allocated(lm._device) / 1024**2}"
                            )
                            break

                        clear_temp_variable(qlayer)
                        _restore_parameters(
                            trainable_parameters,
                            epoch_snapshot,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        retry += 1

                        logger.warning(
                            f"[ROBUST] rollback layer={i} epoch={epochs} "
                            f"retry={retry}/{args.max_retries}: "
                            f"{failure_reason}"
                        )

                        if retry > args.max_retries:
                            logger.warning(
                                f"[ROBUST] SAFE_SKIP layer={i} epoch={epochs}: "
                                f"retries exhausted; retain last finite parameters"
                            )
                            epoch_skipped = True
                            break

                        current_let_lr = max(
                            current_let_lr * args.nan_lr_decay,
                            args.min_lr,
                        )
                        current_lwc_lr = max(
                            current_lwc_lr * args.nan_lr_decay,
                            args.min_lr,
                        )

                        del optimizer
                        optimizer = _build_omni_optimizer(
                            qlayer,
                            use_shift,
                            current_let_lr,
                            current_lwc_lr,
                            args.wd,
                        )

                        # Retries run in true FP32 and do not use scaling.
                        grad_scaler = torch.cuda.amp.GradScaler(enabled=False)

                        next_mode = (
                            "fp32+sanitize"
                            if retry >= 2
                            else "fp32"
                        )
                        logger.warning(
                            f"[ROBUST] FP32 fallback retry with "
                            f"mode={next_mode}, "
                            f"let_lr={current_let_lr:.3e}, "
                            f"lwc_lr={current_lwc_lr:.3e}"
                        )

                    if epoch_skipped:
                        continue

                del optimizer

'''

    # Keep the existing clear_temp_variable(qlayer) and all following code.
    patched = text[:start] + replacement + text[end + len("                del optimizer\n\n"):]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = ROOT.parent / "results" / "source_backups" / stamp
    backup_path = backup_dir / "quantize" / "omniquant.py"
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OMNI, backup_path)

    OMNI.write_text(patched, encoding="utf-8")

    print(f"[backup] {backup_path}")
    print("[ok] robust v2 patch applied")
    print("[info] retry 0=AMP, retry 1=FP32, retry 2+=FP32+gradient sanitization")
    print("[info] exhausted retries now SAFE_SKIP the unstable epoch instead of aborting")


if __name__ == "__main__":
    main()
