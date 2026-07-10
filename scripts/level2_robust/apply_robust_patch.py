#!/usr/bin/env python3
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path('/root/autodl-tmp/omniquant_final/code')
MAIN = ROOT / 'main.py'
OMNI = ROOT / 'quantize' / 'omniquant.py'


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'{label}: expected exactly 1 match, found {count}')
    return text.replace(old, new, 1)


def backup(path: Path, stamp: str) -> None:
    backup_dir = ROOT.parent / "results" / "source_backups" / stamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    relative = path.relative_to(ROOT)
    target = backup_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    print(f'[backup] {target}')


def patch_main(text: str) -> str:
    if '--robust_mode' in text:
        print('[skip] main.py already patched')
        return text
    anchor = '    parser.add_argument("--epochs", type=int, default=10)\n'
    addition = anchor + '''    # Robust low-bit optimization options (course-project extension)\n    parser.add_argument("--robust_mode", action="store_true",\n                        help="enable NaN/Inf recovery, layer rollback and adaptive LR decay")\n    parser.add_argument("--grad_clip", type=float, default=1.0,\n                        help="max gradient norm in robust mode; <=0 disables clipping")\n    parser.add_argument("--max_retries", type=int, default=3,\n                        help="maximum retries for one epoch after non-finite loss/gradient")\n    parser.add_argument("--nan_lr_decay", type=float, default=0.5,\n                        help="multiply LET/LWC learning rates by this value after a failed retry")\n    parser.add_argument("--min_lr", type=float, default=1e-6,\n                        help="minimum learning rate used by robust retry")\n    parser.add_argument("--progress_file", type=str, default=None,\n                        help="optional JSON path recording completed quantized layers")\n'''
    return replace_once(text, anchor, addition, 'main.py argument insertion')


def patch_omni(text: str) -> str:
    if '# === Course-project robust helpers ===' in text:
        print('[skip] omniquant.py already patched')
        return text

    first_import = 'import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton\n'
    if not text.startswith(first_import):
        raise RuntimeError('top-level AutoGPTQ import was not found at file start')
    text = (
        '# AutoGPTQ is only required by --real_quant.\n'
        'qlinear_triton = None\n'
        'qlinear_cuda = None\n'
        + text[len(first_import):]
    )

    old_try = '''try:\n    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda\n    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton\nexcept:\n    print("auto_gptq is required for real quantization")\n'''
    new_try = '''try:\n    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda\n    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton\nexcept Exception:\n    qlinear_cuda = None\n    qlinear_triton = None\n    print("auto_gptq is unavailable; fake quantization is still supported")\n'''
    text = replace_once(text, old_try, new_try, 'AutoGPTQ guarded import')

    helpers = r'''
# === Course-project robust helpers ===
def _build_omni_optimizer(qlayer, use_shift, let_lr, lwc_lr, weight_decay):
    return torch.optim.AdamW(
        [
            {"params": let_parameters(qlayer, use_shift), "lr": let_lr},
            {"params": lwc_parameters(qlayer), "lr": lwc_lr},
        ],
        weight_decay=weight_decay,
    )


def _snapshot_parameters(parameters):
    return [parameter.detach().cpu().clone() for parameter in parameters]


@torch.no_grad()
def _restore_parameters(parameters, snapshot):
    if len(parameters) != len(snapshot):
        raise RuntimeError(
            f"parameter snapshot mismatch: {len(parameters)} vs {len(snapshot)}"
        )
    for parameter, saved in zip(parameters, snapshot):
        parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))


def _atomic_torch_save(payload, final_path):
    final_path = os.path.abspath(final_path)
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    temporary_path = final_path + ".tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, final_path)


def _atomic_json_save(payload, final_path):
    import json
    final_path = os.path.abspath(final_path)
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    temporary_path = final_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary_path, final_path)


def _finite_float(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().item()
    value = float(value)
    return value, math.isfinite(value)
'''
    text = replace_once(text, '\ndef omniquant(\n', helpers + '\ndef omniquant(\n', 'helper insertion')

    text = replace_once(
        text,
        '        omni_parameters = torch.load(args.resume)\n',
        '        omni_parameters = torch.load(args.resume, map_location="cpu", weights_only=False)\n',
        'resume load compatibility',
    )

    old_training = '''        if args.epochs > 0:\n            with torch.no_grad():\n                qlayer.float()      # required for AMP training\n            # create optimizer\n            optimizer = torch.optim.AdamW(\n                [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr}, {"params":lwc_parameters(qlayer),"lr":args.lwc_lr}],weight_decay=args.wd)\n            loss_scaler = utils.NativeScalerWithGradNormCount()\n            \n            for epochs in range(args.epochs):\n                loss_list = []\n                norm_list = []\n                for j in range(args.nsamples//args.batch_size):    \n                    index = j * args.batch_size\n                    # obtain output of quantization model\n                    with traincast():\n                        smooth_and_quant_temporary(qlayer, args, is_llama)\n                        quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]\n                        loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)\n                        if args.aug_loss:\n                            loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)\n                    if not math.isfinite(loss.item()):\n                        logger.info("Loss is NAN, stopping training")\n                        pdb.set_trace()\n                        \n                    loss_list.append(loss.detach().cpu())\n                    optimizer.zero_grad()\n                    norm = loss_scaler(loss, optimizer,parameters= get_omni_parameters(qlayer, use_shift)).cpu()\n                    norm_list.append(norm.data)\n\n                loss_mean = torch.stack(loss_list).mean()\n                norm_mean = torch.stack(norm_list).mean()\n                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")\n            clear_temp_variable(qlayer)\n            del optimizer\n'''

    new_training = '''        if args.epochs > 0:\n            with torch.no_grad():\n                qlayer.float()      # required for AMP training\n\n            if not args.robust_mode:\n                optimizer = torch.optim.AdamW(\n                    [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr},\n                     {"params":lwc_parameters(qlayer),"lr":args.lwc_lr}],\n                    weight_decay=args.wd)\n                loss_scaler = utils.NativeScalerWithGradNormCount()\n\n                for epochs in range(args.epochs):\n                    loss_list = []\n                    norm_list = []\n                    for j in range(args.nsamples//args.batch_size):\n                        index = j * args.batch_size\n                        with traincast():\n                            smooth_and_quant_temporary(qlayer, args, is_llama)\n                            quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]\n                            loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)\n                            if args.aug_loss:\n                                loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)\n                        loss_value, loss_is_finite = _finite_float(loss)\n                        if not loss_is_finite:\n                            raise FloatingPointError(\n                                f"non-finite loss in baseline mode: layer={i}, epoch={epochs}, batch={j}, loss={loss_value}"\n                            )\n                        loss_list.append(loss.detach().cpu())\n                        optimizer.zero_grad()\n                        norm = loss_scaler(\n                            loss, optimizer,\n                            parameters=get_omni_parameters(qlayer, use_shift),\n                        ).cpu()\n                        norm_value, norm_is_finite = _finite_float(norm)\n                        if not norm_is_finite:\n                            raise FloatingPointError(\n                                f"non-finite gradient norm in baseline mode: layer={i}, epoch={epochs}, batch={j}, norm={norm_value}"\n                            )\n                        norm_list.append(norm.data)\n\n                    loss_mean = torch.stack(loss_list).mean()\n                    norm_mean = torch.stack(norm_list).mean()\n                    logger.info(\n                        f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} "\n                        f"max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2}"\n                    )\n                del optimizer\n\n            else:\n                trainable_parameters = list(get_omni_parameters(qlayer, use_shift))\n                if not trainable_parameters:\n                    raise RuntimeError("robust_mode found no learnable LET/LWC parameters")\n\n                current_let_lr = float(args.let_lr)\n                current_lwc_lr = float(args.lwc_lr)\n                optimizer = _build_omni_optimizer(\n                    qlayer, use_shift, current_let_lr, current_lwc_lr, args.wd\n                )\n                grad_scaler = torch.cuda.amp.GradScaler(enabled=not args.deactive_amp)\n\n                for epochs in range(args.epochs):\n                    retry = 0\n                    while True:\n                        epoch_snapshot = _snapshot_parameters(trainable_parameters)\n                        loss_list = []\n                        norm_list = []\n                        failure_reason = None\n\n                        for j in range(args.nsamples//args.batch_size):\n                            index = j * args.batch_size\n                            optimizer.zero_grad(set_to_none=True)\n\n                            with traincast():\n                                smooth_and_quant_temporary(qlayer, args, is_llama)\n                                quant_out = qlayer(\n                                    quant_inps[index:index+args.batch_size,],\n                                    attention_mask=attention_mask_batch,\n                                    position_ids=position_ids,\n                                )[0]\n                                loss = loss_func(\n                                    fp_inps[index:index+args.batch_size,], quant_out\n                                )\n                                if args.aug_loss:\n                                    loss += loss_func(\n                                        fp_inps_2[index:index+args.batch_size,], quant_out\n                                    )\n\n                            loss_value, loss_is_finite = _finite_float(loss)\n                            if not loss_is_finite:\n                                failure_reason = f"non-finite loss={loss_value} at batch={j}"\n                                break\n\n                            grad_scaler.scale(loss).backward()\n                            grad_scaler.unscale_(optimizer)\n                            max_norm = float(args.grad_clip) if args.grad_clip > 0 else float("inf")\n                            grad_norm = torch.nn.utils.clip_grad_norm_(\n                                trainable_parameters, max_norm=max_norm\n                            )\n                            norm_value, norm_is_finite = _finite_float(grad_norm)\n                            if not norm_is_finite:\n                                failure_reason = f"non-finite grad_norm={norm_value} at batch={j}"\n                                optimizer.zero_grad(set_to_none=True)\n                                break\n\n                            grad_scaler.step(optimizer)\n                            grad_scaler.update()\n                            loss_list.append(loss.detach().cpu())\n                            norm_list.append(torch.tensor(norm_value, dtype=torch.float32))\n\n                        if failure_reason is None:\n                            loss_mean = torch.stack(loss_list).mean()\n                            norm_mean = torch.stack(norm_list).mean()\n                            logger.info(\n                                f"[ROBUST] layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} "\n                                f"retry:{retry} let_lr:{current_let_lr:.3e} lwc_lr:{current_lwc_lr:.3e} "\n                                f"max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2}"\n                            )\n                            break\n\n                        clear_temp_variable(qlayer)\n                        _restore_parameters(trainable_parameters, epoch_snapshot)\n                        optimizer.zero_grad(set_to_none=True)\n                        retry += 1\n                        logger.warning(\n                            f"[ROBUST] rollback layer={i} epoch={epochs} "\n                            f"retry={retry}/{args.max_retries}: {failure_reason}"\n                        )\n                        if retry > args.max_retries:\n                            raise FloatingPointError(\n                                f"robust retries exhausted at layer={i}, epoch={epochs}: {failure_reason}"\n                            )\n\n                        current_let_lr = max(current_let_lr * args.nan_lr_decay, args.min_lr)\n                        current_lwc_lr = max(current_lwc_lr * args.nan_lr_decay, args.min_lr)\n                        del optimizer\n                        optimizer = _build_omni_optimizer(\n                            qlayer, use_shift, current_let_lr, current_lwc_lr, args.wd\n                        )\n                        grad_scaler = torch.cuda.amp.GradScaler(enabled=not args.deactive_amp)\n                        logger.warning(\n                            f"[ROBUST] retry with let_lr={current_let_lr:.3e}, "\n                            f"lwc_lr={current_lwc_lr:.3e}"\n                        )\n\n                del optimizer\n\n            clear_temp_variable(qlayer)\n'''
    training_start_marker = (
        '        if args.epochs > 0:\n'
        '            with torch.no_grad():\n'
        '                qlayer.float()'
    )
    training_end_marker = '        qlayer.half()'
    training_start = text.find(training_start_marker)
    training_end = text.find(training_end_marker, training_start)
    if training_start < 0 or training_end < 0:
        raise RuntimeError(
            'training-loop anchors were not found in quantize/omniquant.py'
        )
    text = text[:training_start] + new_training + text[training_end:]

    old_save = '''            omni_parameters[i] = omni_state_dict(qlayer)\n            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))\n'''
    new_save = '''            omni_parameters[i] = omni_state_dict(qlayer)\n            checkpoint_path = os.path.join(args.output_dir, "omni_parameters.pth")\n            _atomic_torch_save(omni_parameters, checkpoint_path)\n\n            progress_path = (\n                args.progress_file\n                if args.progress_file\n                else os.path.join(args.output_dir, "quant_progress.json")\n            )\n            completed_layers = sorted(int(key) for key in omni_parameters.keys())\n            _atomic_json_save(\n                {\n                    "completed_layers": completed_layers,\n                    "last_completed_layer": i,\n                    "num_layers": len(layers),\n                    "complete": len(completed_layers) == len(layers),\n                    "robust_mode": bool(args.robust_mode),\n                    "wbits": int(args.wbits),\n                    "abits": int(args.abits),\n                    "group_size": args.group_size,\n                    "epochs": int(args.epochs),\n                },\n                progress_path,\n            )\n'''
    text = replace_once(text, old_save, new_save, 'atomic checkpoint replacement')

    old_real = '''        if args.real_quant:\n            assert args.wbits in [2,3,4] and args.abits >= 16   # only support weight-only quantization\n'''
    new_real = '''        if args.real_quant:\n            if qlinear_triton is None or qlinear_cuda is None:\n                raise ImportError("--real_quant requires a working AutoGPTQ installation")\n            assert args.wbits in [2,3,4] and args.abits >= 16   # only support weight-only quantization\n'''
    text = replace_once(text, old_real, new_real, 'real-quant dependency guard')
    return text


def main() -> None:
    if not MAIN.is_file() or not OMNI.is_file():
        raise FileNotFoundError(f'Expected source tree at {ROOT}')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup(MAIN, stamp)
    backup(OMNI, stamp)
    MAIN.write_text(patch_main(MAIN.read_text(encoding='utf-8')), encoding='utf-8')
    OMNI.write_text(patch_omni(OMNI.read_text(encoding='utf-8')), encoding='utf-8')
    print('[ok] robust patch applied')


if __name__ == '__main__':
    main()
