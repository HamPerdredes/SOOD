#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/18 17:03
# @Author : WeiHua

from mmrotate.models import ROTATED_DETECTORS, RotatedBaseDetector, build_loss


@ROTATED_DETECTORS.register_module()
class RotatedSemiDetector(RotatedBaseDetector):
    def __init__(self, model: dict, semi_loss, train_cfg=None, test_cfg=None):
        super(RotatedSemiDetector, self).__init__()
        self.submodules = list(model.keys())
        for k, v in model.items():
            setattr(self, k, v)
        self.semi_loss = build_loss(semi_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.inference_on = self.test_cfg.get("inference_on", self.submodules[0])

    def model(self, **kwargs):
        if "submodule" in kwargs:
            assert (
                    kwargs["submodule"] in self.submodules
            ), "Detector does not contain submodule {}".format(kwargs["submodule"])
            model = getattr(self, kwargs["submodule"])
        else:
            model = getattr(self, self.inference_on)
        return model

    def freeze(self, model_ref: str):
        assert model_ref in self.submodules
        model = getattr(self, model_ref)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def forward_test(self, imgs, img_metas, **kwargs):
        return self.model(**kwargs).forward_test(imgs, img_metas, **kwargs)

    async def aforward_test(self, *, img, img_metas, **kwargs):
        return self.model(**kwargs).aforward_test(img, img_metas, **kwargs)

    def extract_feat(self, imgs):
        return self.model().extract_feat(imgs)

    async def aforward_test(self, *, img, img_metas, **kwargs):
        return self.model(**kwargs).aforward_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.model(**kwargs).aug_test(imgs, img_metas, **kwargs)

    def simple_test(self, img, img_metas, **kwargs):
        return self.model(**kwargs).simple_test(img, img_metas, **kwargs)

    async def async_simple_test(self, img, img_metas, **kwargs):
        return self.model(**kwargs).async_simple_test(img, img_metas, **kwargs)

    def show_result(self, *args, **kwargs):
        self.model().CLASSES = self.CLASSES
        return self.model().show_result(*args, **kwargs)
