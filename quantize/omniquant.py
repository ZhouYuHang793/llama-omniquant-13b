# AutoGPTQ is only required by --real_quant.
qlinear_triton = None
qlinear_cuda = None
import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
from quantize.utils import let_parameters, lwc_parameters, get_omni_parameters,\
                            omni_state_dict, register_scales_and_zeros,smooth_and_quant_temporary,\
                            smooth_and_quant_inplace,clear_temp_variable,set_quant_state
try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except Exception:
    qlinear_cuda = None
    qlinear_triton = None
    print("auto_gptq is unavailable; fake quantization is still supported")



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     

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

def omniquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.net.lower():
        is_llama = True   # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    
    
    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower() or "mixtral" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    
    attention_mask = cache["attention_mask"]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None



    if args.resume:
        omni_parameters = torch.load(args.resume, map_location="cpu", weights_only=False)
    else:
        omni_parameters = {}

    
    
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        if "mixtral" in args.net.lower():  
            # for mixtral, we only leverage lwc, which can be achieve by simply replace Linear with QuantLinear
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear) and not "gate" in name:       # do not quantize gate
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)    
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)

        
        # obtain output of full-precision model
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
                        if args.aug_loss:
                            fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
        # init smooth parameters
        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        if is_llama or args.abits == 16:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
        if args.let:
            # init channel-wise scaling and shift
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
                                
        if args.resume:
            qlayer.load_state_dict(omni_parameters[i], strict=False)
        

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training

            if not args.robust_mode:
                optimizer = torch.optim.AdamW(
                    [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr},
                     {"params":lwc_parameters(qlayer),"lr":args.lwc_lr}],
                    weight_decay=args.wd)
                loss_scaler = utils.NativeScalerWithGradNormCount()

                for epochs in range(args.epochs):
                    loss_list = []
                    norm_list = []
                    for j in range(args.nsamples//args.batch_size):
                        index = j * args.batch_size
                        with traincast():
                            smooth_and_quant_temporary(qlayer, args, is_llama)
                            quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                            loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                            if args.aug_loss:
                                loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)
                        loss_value, loss_is_finite = _finite_float(loss)
                        if not loss_is_finite:
                            raise FloatingPointError(
                                f"non-finite loss in baseline mode: layer={i}, epoch={epochs}, batch={j}, loss={loss_value}"
                            )
                        loss_list.append(loss.detach().cpu())
                        optimizer.zero_grad()
                        norm = loss_scaler(
                            loss, optimizer,
                            parameters=get_omni_parameters(qlayer, use_shift),
                        ).cpu()
                        norm_value, norm_is_finite = _finite_float(norm)
                        if not norm_is_finite:
                            raise FloatingPointError(
                                f"non-finite gradient norm in baseline mode: layer={i}, epoch={epochs}, batch={j}, norm={norm_value}"
                            )
                        norm_list.append(norm.data)

                    loss_mean = torch.stack(loss_list).mean()
                    norm_mean = torch.stack(norm_list).mean()
                    logger.info(
                        f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} "
                        f"max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2}"
                    )
                del optimizer

            else:
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

            clear_temp_variable(qlayer)
        qlayer.half() 
        # real smooth and quantization
        smooth_and_quant_inplace(qlayer, args, is_llama)
        if args.epochs>0:
            # update input of quantization model
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                with traincast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer)
            checkpoint_path = os.path.join(args.output_dir, "omni_parameters.pth")
            _atomic_torch_save(omni_parameters, checkpoint_path)

            progress_path = (
                args.progress_file
                if args.progress_file
                else os.path.join(args.output_dir, "quant_progress.json")
            )
            completed_layers = sorted(int(key) for key in omni_parameters.keys())
            _atomic_json_save(
                {
                    "completed_layers": completed_layers,
                    "last_completed_layer": i,
                    "num_layers": len(layers),
                    "complete": len(completed_layers) == len(layers),
                    "robust_mode": bool(args.robust_mode),
                    "wbits": int(args.wbits),
                    "abits": int(args.abits),
                    "group_size": args.group_size,
                    "epochs": int(args.epochs),
                },
                progress_path,
            )
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
        if args.real_quant:
            if qlinear_triton is None or qlinear_cuda is None:
                raise ImportError("--real_quant requires a working AutoGPTQ installation")
            assert args.wbits in [2,3,4] and args.abits >= 16   # only support weight-only quantization
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                if args.wbits == 3:
                    q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                else:
                    q_linear = qlinear_triton.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer, q_linear)       
                print(f"pack quantized {name} finished")
                del module        
        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

