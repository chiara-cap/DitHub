# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
import random
import torch
import torch.nn.functional as F

from torch import nn
from pathlib import Path
from typing import List

from groundingdino.util import get_tokenlizer
from groundingdino.util.misc import NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list
from groundingdino.util.task_memory import TaskMemory

from ..registry import MODULE_BUILD_FUNCS
from .backbone import build_backbone
from .bertwarper import BertModelWarper, generate_masks_with_special_tokens_and_transfer_map
from .transformer_for_adapter import build_transformer
from .utils import MLP, ContrastiveEmbed, recover_to_cls_logits
from .criterion import build_criterion
from ...util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances

zero_value = 1e-8

class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
        criterion=None,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],
        pixel_std: List[float] = [123.675, 116.280, 103.530],
        device="cuda",
        select_box_nums_for_evaluation=200,
        # pat
        freeze_all=False,
        num_select_prompt=200,
        use_add_names=False,
        use_learned_names=False,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = max_text_len
        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)
        
        # bert text feature maper
        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        
        # learned classes
        self.learned_classes = []
        
        # for tuning
        self.prompt_memory_pool = nn.ParameterDict()
        self.num_select_prompt = num_select_prompt
        self.use_learned_names = use_learned_names    

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed(max_text_len=self.max_text_len)

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
            
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None
        
        # critetion
        self.criterion = criterion
        
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        
        #  device
        self.device = device
        
        # init param
        self._reset_parameters()
        
        #  add names
        self.use_add_names = use_add_names
        
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation
        
        # freeze
        self.freeze_all = freeze_all

        # LoRA
        self.is_custom_attention = False
        self.is_lora_applied = False

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    def forward(self, batched_inputs, **kw):
        # process images
        images = self.preprocess_image(batched_inputs)
        assert isinstance(images, ImageList)
        samples = nested_tensor_from_tensor_list(images)

        # prepare captions
        captions = [x["captions"] for x in batched_inputs]
        names_list = [x["captions"][:-1].split(".") for x in batched_inputs]

        # update task memory classes
        memory_classes = []
        for batch_elem, names_elem in zip(batched_inputs, names_list):
            if 'instances' not in batch_elem:
                names_elem = [f'class_{n[0].lower() + n[1:]}' for n in names_elem]
                memory_classes = names_elem
                break
            elem_classes = batch_elem['instances'].gt_classes.tolist()

            if not elem_classes:
                elem_classes = [0]

            curr_dataset = Path(batch_elem['file_name']).parts[2].lower()
            curr_class_int = random.choice(elem_classes)
            curr_class_str = TaskMemory().task_mapping[curr_dataset][curr_class_int]
            curr_class_str = curr_class_str[0].lower() +  curr_class_str[1:]
            memory_classes.append(f'class_{curr_class_str}')

        TaskMemory().set_classes(memory_classes)
        
        if self.training:
            TaskMemory().add_counter(memory_classes)

        if (self.use_add_names and (not self.training)) or (self.use_learned_names and self.training):
            non_overlap_classes = [class_name for class_name in \
                self.learned_classes if (class_name not in names_list[0])]
            if self.training and (len(non_overlap_classes) >= self.num_select_prompt):
                    non_overlap_classes = random.sample(non_overlap_classes, self.num_select_prompt)    
            for idx, (caption, names) in enumerate(zip(captions, names_list)):
                names_list[idx] = names + non_overlap_classes
                captions[idx] = caption + ".".join(non_overlap_classes)
                if not captions[idx].endswith("."): captions[idx] = captions[idx] + "."

        tokenized = self.tokenizer(captions, padding="longest",
                                   return_tensors="pt").to(samples.device)
        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        # prepare targets
        targets = None
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, cate_to_token_mask_list, names_list)

        # encoder texts
        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768
        
        encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
        
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195
        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len]

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }

        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, _ = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict)

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # output
        outputs_class = torch.stack(
            [
                recover_to_cls_logits(layer_cls_embed(layer_hs, text_dict), \
                    cate_to_token_mask_list, for_fill=-100.0)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ])
        
        out = {"pred_logits": outputs_class[-1],
               "pred_boxes": outputs_coord_list[-1],
               "cate_to_token_mask_list": cate_to_token_mask_list}


        if self.training:
            # for intermediate outputs
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)

            # for encoder output
            if hs_enc is not None:
                # prepare intermediate outputs
                interm_coord = ref_enc[-1]
                interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
                interm_class = recover_to_cls_logits(interm_class, cate_to_token_mask_list, for_fill=-100.0)
                out["enc_outputs"] = {"pred_logits": interm_class, "pred_boxes": interm_coord}
                
            # return loss
            assert targets is not None
            assert self.criterion is not None
            loss_dict = self.criterion(out, targets, dn_meta)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = out["pred_logits"]
            box_pred = out["pred_boxes"]
            results = self.dt_inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def prepare_targets(self, targets, cate_to_token_mask_list, names_list):
        new_targets = []
        for targets_per_image, _, _ in \
            zip(targets, cate_to_token_mask_list, names_list):
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def dt_inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # box_cls.shape: 1, 300, 80
        # box_pred.shape: 1, 300, 4
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def normalizer(self, x):
        pixel_mean = torch.Tensor(self.pixel_mean).to(x.device).view(3, 1, 1)
        pixel_std = torch.Tensor(self.pixel_std).to(x.device).view(3, 1, 1)
        return (x - pixel_mean) / pixel_std

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def unfreeze_module_(self, pat_names):
        for module_name, param in self.named_parameters():
            for pat_name in pat_names:
                if pat_name in module_name:
                    param.requires_grad = True
                    print("unfreeze:", module_name)
                    break
    
    def load_state_dict(self, state_dict, strict=True):
        for k, v in state_dict.items():
            if "prompt_memory_pool" in k:
                class_name = k.split(".")[-1]
                class_name = class_name[1:-1]
                self.learned_classes.append(class_name)
                print("learned class:", class_name)
        res = super().load_state_dict(state_dict=state_dict, strict=strict)
        
        if self.freeze_all:
            for param in self.parameters():
                param.requires_grad = False
        return res

    def load_custom_attention(self):
        # iterate over each decoder layer (and QIM) and change the self-attention (nn.MultiheadAttention) to our custom attention module
        # this is because we implement the qkv projection as nn.Linear instead as of functional calls
        from groundingdino.util.lora_utils import apply_custom_attention
        for layer in self.transformer.decoder.layers:
            apply_custom_attention(layer, ['self_attn', 'ca_text'])

        for layer in self.transformer.encoder.text_layers:
            apply_custom_attention(layer, ['self_attn'])
    
    @torch.no_grad()
    def inference(self, samples: NestedTensor, targets: List = None, **kw):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if targets is None:
            captions = kw["captions"]
        else:
            captions = [t["caption"] for t in targets]
        names_list = captions[0][:-1].split(".")
        names_list = [names_list]

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        if (self.use_add_names and (not self.training)) or (self.use_learned_names and self.training):
            non_overlap_classes = [class_name for class_name in \
                self.learned_classes if (class_name not in names_list[0])]
            if self.training and (len(non_overlap_classes) >= self.num_select_prompt):
                    non_overlap_classes = random.sample(non_overlap_classes, self.num_select_prompt)    
            for idx, (caption, names) in enumerate(zip(captions, names_list)):
                names_list[idx] = names + non_overlap_classes
                captions[idx] = caption + ".".join(non_overlap_classes)
                if not captions[idx].endswith("."): captions[idx] = captions[idx] + "."

        tokenized = self.tokenizer(captions, padding="longest",
                                   return_tensors="pt").to(samples.device)
        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        # prepare targets
        targets = None

        # encoder texts
        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768
        
        encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
        
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195
        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len]

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }

        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = None
        hs, reference, _, _, _ = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict)

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # output
        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1], "cate_to_token_mask_list": cate_to_token_mask_list}
        return out

@MODULE_BUILD_FUNCS.registe_with_name(module_name="dtgroundingdino")
def build_dt_groundingdino(cfg_model):

    backbone = build_backbone(cfg_model)
    transformer = build_transformer(cfg_model)
    criterion = build_criterion(cfg_model)

    dn_labelbook_size = cfg_model.dn_labelbook_size
    dec_pred_bbox_embed_share = cfg_model.dec_pred_bbox_embed_share
    sub_sentence_present = cfg_model.sub_sentence_present
    
    if hasattr(cfg_model, "use_learned_names"):
        use_learned_names = cfg_model.use_learned_names
    else:
        use_learned_names = False

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=cfg_model.num_queries,
        aux_loss=cfg_model.aux_loss,
        iter_update=True,
        query_dim=4,
        num_feature_levels=cfg_model.num_feature_levels,
        nheads=cfg_model.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=cfg_model.two_stage_type,
        two_stage_bbox_embed_share=cfg_model.two_stage_bbox_embed_share,
        two_stage_class_embed_share=cfg_model.two_stage_class_embed_share,
        num_patterns=cfg_model.num_patterns,
        dn_number=0,
        dn_box_noise_scale=cfg_model.dn_box_noise_scale,
        dn_label_noise_ratio=cfg_model.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=cfg_model.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=cfg_model.max_text_len,
        criterion=criterion,
        freeze_all=cfg_model.freeze_all,
        select_box_nums_for_evaluation=cfg_model.select_box_nums_for_evaluation,
        use_learned_names=use_learned_names,
    )

    return model
