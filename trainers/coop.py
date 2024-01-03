import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.autograd import Variable

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from helpers import distillation
import copy

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class FeatureLoss(nn.Module):
    def __init__(self,):
        super(FeatureLoss, self).__init__()

    def forward(self, text_features, fix_label_features, label):

        loss_num = 0
        loss = torch.tensor(0.).to(text_features.device)

        for i in range(label.size(0)):
            if torch.sum(label !=  label[i]) > 0:
                index = label[i]    
                dist_ap = (1- F.cosine_similarity(text_features[index].unsqueeze(0), fix_label_features[index].unsqueeze(0).detach()).squeeze())
                loss_num += 1
                loss += dist_ap
                    
        if loss_num != 0:
            return loss / loss_num
        else:
            return loss
        
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        origin_clip = cfg.origin_clip
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        if origin_clip:
            self.ctx.requires_grad = False
            
        self.random_init = cfg.random_init

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        
        self.name_lens = name_lens
        self.min_len = min(self.name_lens) # 1
        if self.min_len > 1:
            print("origin len is ", name_lens)
            classnames = self.revise_classnames(classnames, name_lens, self.min_len)
            name_lens = [len(_tokenizer.encode(name)) for name in classnames] 
            print("later len is ", name_lens)
            
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.eval_only = cfg.eval_only
        self.n_cls = n_cls
        print(f"Number of classes: {n_cls}")
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        self.base = (cfg.DATASET.SUBSAMPLE_CLASSES == "base" or cfg.DATASET.SUBSAMPLE_CLASSES == "all" )
        
        self._init_suffix_dict(classnames, clip_model, dtype)
        self._get_token_classes(dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.base or not self.eval_only:
            self.register_buffer("token_suffix", embedding[:, 1 + self.min_len + n_ctx:, :])  # EOS
            self.register_buffer("token_suffix_test", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        
    def revise_classnames(self, classnames, name_lens, min_len):
        if min(name_lens) < min_len:
            for i in range(len(classnames)):
                if name_lens[i] < min_len:
                    classnames[i] = ("<|startoftext|> "*(min_len - name_lens[i])) + classnames[i]
        return classnames        
    
    def _init_suffix_dict(self, classnames, clip_model, dtype):
            
        self.suffix_classes = {}
        for name in classnames:
            self.suffix_classes[name] = clip_model.token_embedding(clip.tokenize(name)).type(dtype)
    
    def _get_token_classes(self, dtype):
            
        if self.base or not self.eval_only:
            self.token_classes_all = torch.cat([self.suffix_classes[name] for name in self.suffix_classes]).type(dtype)            
            self.token_classes = self.token_classes_all[:, 1:self.min_len+1, :]
            if self.random_init:
                nn.init.normal_(self.token_classes, std=0.02)
            self.token_classes = nn.Parameter(self.token_classes)
            self.fix_token = copy.deepcopy(self.token_classes)
            self.fix_token.requires_grad = False
        else:
            pass
        
    def construct_prompts(self, ctx, prefix, suffix):
        
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        else:
            raise ValueError
        
        return prompts

    def construct_prompts_v2(self, ctx, prefix, classes, suffix):
        
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    classes, # (dim0, 1, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        else:
            raise ValueError
        
        return prompts
             
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.base or not self.eval_only:
            classes = self.token_classes
            prompts = self.construct_prompts_v2(ctx, prefix, classes, suffix)
        else:
            prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts

    def forward_test(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix_test
    
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.triplet_loss = cfg.triplet_loss
        self.shots = cfg.shots
        self.loss_feature_fn = FeatureLoss()
        
    def _get_origin_feature(self):
        tokenized_prompts = self.tokenized_prompts
        prompts = self.prompt_learner.forward_test()
        fix_label_features = self.text_encoder(prompts, tokenized_prompts)
        
        return fix_label_features
    
    def forward(self, image, label=None):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            if self.triplet_loss:
                
                fix_label_features = self._get_origin_feature()
                fix_label_features = fix_label_features / fix_label_features.norm(dim=-1, keepdim=True)
                
                fix_logists = logit_scale * image_features @ fix_label_features.t()
                loss_logits = distillation(logits, fix_logists, T=2.0)
                
                loss_feature = self.loss_feature_fn(text_features, fix_label_features, label)
                loss_paramter = F.mse_loss(self.prompt_learner.token_classes, self.prompt_learner.fix_token.detach(), reduction="sum")
                
                entorpy_loss = F.cross_entropy(logits, label)
                
                return entorpy_loss + 1 / self.shots * loss_paramter + loss_feature + 0.05 * loss_logits
            else:
                return F.cross_entropy(logits, label)

        return logits


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        
        prec = self.cfg.TRAINER.COOP.PREC
        
        if prec == "amp":
            with autocast():
                loss = model(image, label) 
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss = model(image, label)
            self.model_backward_and_update(loss)
        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]
                
                
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
