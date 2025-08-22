import torch
import numpy as np

class DepthPixelMetric:
    def __init__(
        self,
        threshold=1.25,
        depth_cap=10,
        aligned_by_depth=False,
        aligned_by_median=False,
    ):
        self.__threshold = threshold
        self.__depth_cap = depth_cap
        self.__aligned_by_depth = aligned_by_depth
        self.__aligned_by_median = aligned_by_median
        self.__trim = 0.2

    def compute_errors(self, gt, pred, mask):
        gt = gt[mask == 1.0]
        pred = pred[mask == 1.0]

        thresh = np.maximum((gt / pred), (pred / gt))
        d1 = (thresh < 1.25).mean()
        d2 = (thresh < 1.25**2).mean()
        d3 = (thresh < 1.25**3).mean()

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        err = np.log(pred) - np.log(gt)
        silog = np.sqrt(np.mean(err**2) - np.mean(err) ** 2) * 100

        err = np.abs(np.log10(pred) - np.log10(gt))
        log10 = np.mean(err)

        return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3

    def compute_scale_and_shift(self, prediction, target, mask):

        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        a_00 = torch.sum(mask * prediction * prediction, (1, 2))
        a_01 = torch.sum(mask * prediction, (1, 2))
        a_11 = torch.sum(mask, (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = torch.sum(mask * prediction * target, (1, 2))
        b_1 = torch.sum(mask * target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = torch.zeros_like(b_0)
        x_1 = torch.zeros_like(b_1)

        det = a_00 * a_11 - a_01 * a_01
        # A needs to be a positive definite matrix.
        valid = det > 0

        x_0[valid] = (
            a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]
        ) / det[valid]
        x_1[valid] = (
            -a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]
        ) / det[valid]

        return x_0, x_1

    def __call__(self, prediction, target, mask):
        # transform predicted disparity to aligned depth
        if self.__aligned_by_depth:
            if self.__aligned_by_median:
                scale = torch.median(
                    target[mask.bool()] / prediction[mask.bool()]
                )
                prediciton_depth = (
                    scale.view(-1, 1, 1) * prediction
                )  # + shift.view(-1, 1, 1)
            else:
                scale, shift = self.compute_scale_and_shift(
                    prediction, target, mask
                )
                prediciton_depth = scale.view(
                    -1, 1, 1
                ) * prediction + shift.view(-1, 1, 1)

                re_scale_mask = mask
                for _ in range(2):
                    err_map = torch.abs(prediciton_depth - target) * mask
                    thr, _ = torch.sort(err_map.view(-1))
                    thr = thr[(re_scale_mask == 0).sum() :]
                    thr = thr[int(0.8 * thr.shape[0])]
                    err_mask = (err_map < thr).float()
                    re_scale_mask = err_mask * re_scale_mask
                    scale, shift = self.compute_scale_and_shift(
                        prediction, target, re_scale_mask
                    )
                    prediciton_depth = scale.view(
                        -1, 1, 1
                    ) * prediction + shift.view(-1, 1, 1)

            prediciton_depth[
                prediciton_depth > self.__depth_cap
            ] = self.__depth_cap
        else:
            target_disparity = torch.zeros_like(target)
            target_disparity[mask == 1] = 1.0 / target[mask == 1]
            if self.__aligned_by_median:
                scale = torch.median(
                    target[mask.bool()] / prediction[mask.bool()]
                )
                prediction_aligned = (
                    scale.view(-1, 1, 1) * prediction
                )  # + shift.view(-1, 1, 1)
            else:
                scale, shift = self.compute_scale_and_shift(
                    prediction, target_disparity, mask
                )
                prediction_aligned = scale.view(
                    -1, 1, 1
                ) * prediction + shift.view(-1, 1, 1)

                # re_compute_scale
                re_scale_mask = mask
                for _ in range(2):
                    error_map = (prediction_aligned - target_disparity).abs()
                    res = error_map[mask == 1]
                    trimmed, _ = torch.sort(res.view(-1), descending=False)
                    thr = trimmed[int(len(res) * (1.0 - self.__trim))]

                    err_mask = (error_map < thr).float()
                    re_scale_mask = err_mask * re_scale_mask
                    scale, shift = self.compute_scale_and_shift(
                        prediction, target_disparity, re_scale_mask
                    )
                    prediction_aligned = scale.view(
                        -1, 1, 1
                    ) * prediction + shift.view(-1, 1, 1)

            disparity_cap = 1.0 / self.__depth_cap
            prediction_aligned[
                prediction_aligned < disparity_cap
            ] = disparity_cap
            prediciton_depth = 1.0 / prediction_aligned

        # bad pixel
        err = self.compute_errors(
            gt=target.cpu().numpy(),
            pred=prediciton_depth.cpu().numpy(),
            mask=mask.cpu().numpy(),
        )
        err_map = (prediciton_depth - target).abs()
        err_map[mask == 0] = 0

        return err, err_map, prediciton_depth
