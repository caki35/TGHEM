import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import gudhi as gd
import gudhi.wasserstein



class BinaryDiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.
    """

    def __init__(self, ignore_index=None, batch_dice=None, use_sigmoid=True, reduction='mean', **kwargs):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = 1  # suggest set a large number when target area is large,like '10|100'
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.use_sigmoid = use_sigmoid
        self.batch_dice = False  # treat a large map when True
        if 'batch_loss' in kwargs.keys():
            self.batch_dice = kwargs['batch_loss']

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"
        if self.use_sigmoid:
            output = torch.sigmoid(output)

        if self.ignore_index is not None:
            validmask = (target != self.ignore_index).float()
            output = output.mul(validmask)  # can not use inplace for bp
            target = target.float().mul(validmask)

        dim0 = output.shape[0]
        if self.batch_dice:
            dim0 = 1

        output = output.contiguous().view(dim0, -1)
        target = target.contiguous().view(dim0, -1).float()

        num = 2 * torch.sum(torch.mul(output, target), dim=1) + self.smooth
        den = torch.sum(output.abs() + target.abs(), dim=1) + self.smooth

        loss = 1 - (num / den)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class SegmentationLoss(nn.Module):
    """Combined Dice + BCE Loss with Online Hard Example Mining (OHEM) for binary segmentation."""
    def __init__(self, thresh=None, min_kept=100000, ignore_index=None,
                 dice_weight=0.5, bce_weight=0.5, reduction='mean'):
        """
        Args:
            thresh (float, optional): Confidence threshold below which predictions are considered hard examples:contentReference[oaicite:6]{index=6}.
                                       If None, selects top `min_kept` hardest pixels by loss:contentReference[oaicite:7]{index=7}.
            min_kept (int): Minimum number of pixels to keep per batch (for threshold mode, also used to adjust threshold):contentReference[oaicite:8]{index=8}.
            ignore_index (int, optional): Label value to ignore in loss computation (e.g., 255 for segmentation mask ignore).
            dice_weight (float): Weight for the Dice loss component.
            bce_weight (float): Weight for the BCE loss component.
            reduction (str): none -> calculate per-iamge, mean and sum -> across whole batch
        """
        super().__init__()
        assert min_kept > 1, "`min_kept` should be > 1."
        assert reduction in ('mean', 'sum', 'none')
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index 
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.reduction = reduction # none -> calculate per-iamge, mean and sum -> across whole batch
        # Dice loss can be an external implementation. Here we assume a BinaryDiceLoss is defined elsewhere.
        # If not, one could implement a simple dice loss inside this class.
        self.dice_loss_fn = BinaryDiceLoss(reduction=reduction)

    def _per_image_topk_mask(self, loss_map, valid_mask):
        """
        loss_map: a per-pixel loss (B,H,W) 
        valid_mask: a boolean valid_mask (B,H,W)
            if ignore_index is None, its full of 1s
        Select top min_kept per image. Returns float mask (B,H,W) in {0,1}.
        """
        B, H, W = loss_map.shape
        mask = torch.zeros_like(loss_map, dtype=loss_map.dtype)
        # flatten per image
        lm = loss_map.view(B, -1)
        vm = valid_mask.view(B, -1)
        # iterate across images in mini-batch
        for b in range(B):
            v = vm[b]
            n_valid = int(v.sum().item())
            if n_valid == 0:
                continue
            k = min(self.min_kept, n_valid)
            # take top-k hardest among valid pixels
            scores = lm[b].masked_fill(~v, -1e30) # consider pixels that only exists in valid mask
            topv, topi = torch.topk(scores, k=k, largest=True, sorted=False)
            # Build a (H,W) mask with 1s at those k positions, 0 elsewhere.
            mb = torch.zeros_like(v, dtype=loss_map.dtype)
            mb.scatter_(0, topi, 1.0)
            mask[b] = mb.view(H, W)
        return mask

    def _per_image_thresh_mask(self, prob, target_mask, valid_mask):
        """
        prob: per-pixel probability (B,H,W) (after sigmoid)
        valid_mask: a boolean valid_mask (B,H,W)
            if ignore_index is None, its full of 1s
        target_mask: the ground-truth binary mask (B,H,W)
        
        Confidence-threshold OHEM per image.
        Keeps at least min_kept pixels per image by adapting threshold up.
        Returns float mask (B,H,W) in {0,1}.
        """
        B, H, W = prob.shape
        mask = torch.zeros_like(prob, dtype=prob.dtype)
        # computes the model’s confidence in the ground-truth class at each pixel.
        #  per-pixel probability assigned to the correct class.
        conf = torch.where(target_mask.float() > 0.5, prob, 1 - prob) # Return a tensor of elements selected from either prob or 1-prob, depending on target_mask.
        conf = conf.masked_fill(~valid_mask, 1.0)  # ignored -> high conf (won't be selected)

        # iterate across images in mini-batch
        for b in range(B):
            c  = conf[b][valid_mask[b]]
            if c.numel() == 0:
                continue
            c_sorted, _ = torch.sort(c, descending=False)  # low conf = hard
            
            #ensure that self.min_kept does not exceed valid max
            k = min(self.min_kept, c_sorted.numel() - 1)
            min_conf = c_sorted[k]
            
            # Guarantees at least min_kept pixel to be selected per image
            # if self.thresh can not select enough pixel(topK), increase threshold to the index of topK
            eff_thresh = max(min_conf.item(), self.thresh if self.thresh is not None else 0.0)
            # keep pixel under a confidence
            mb = (conf[b] < eff_thresh) & valid_mask[b]
            # the per-image selection mask (B,H,W).
            mask[b] = mb.to(mask.dtype)
        return mask

    def forward(self, pred_logits, target):
        """
        Compute the combined Dice + OHEM-BCE loss.
        Args:
            pred_logits (Tensor): Predicted logits of shape (B, 1, H, W).
            target (Tensor): Ground truth mask of shape (B, H, W), with values 0 or 1.
        """
        B, C, H, W = pred_logits.shape
        assert C == 1, "This loss is for binary segmentation with a single-channel prediction."

        # Prepare tensors
        pred_logits = pred_logits.squeeze(1)            # shape (B, H, W)
        target_mask = target.clone()                    # shape (B, H, W)
        valid_mask = torch.ones_like(target_mask, dtype=torch.bool)  # will be mask of pixels to consider
        if self.ignore_index is not None:
            valid_mask = target_mask != self.ignore_index  # ignore pixels where target == ignore_index
            # For ignore pixels, set a valid dummy value in target (e.g., 0) to avoid misuse in computations
            target_mask = target_mask.masked_fill(~valid_mask, 0)

        # Compute the OHEM selection mask (seg_weight) with no gradient, as it’s used for sampling
        with torch.no_grad():
            if self.thresh is not None:
                # Confidence-based hard example selection
                # Compute predicted probabilities for class=1
                prob = torch.sigmoid(pred_logits)  # (B, H, W), probability of class 1
                if self.reduction == 'none': #image-level selection
                    # Guarantees at least min_kept per image (when possible).
                    # (B,H,W) mask in {0,1} based on prediction confidence threshold
                    seg_weight = self._per_image_thresh_mask(prob, target_mask, valid_mask)
                else: #batch-level selection
                    # Compute confidence of correct class for each pixel:
                    # If target=1 -> use prob, if target=0 -> use (1 - prob)
                    target_float = target_mask.float()
                    conf = torch.where(target_float > 0.5, prob, 1 - prob)  # confidence of the correct class for each pixel
                    # Exclude ignore pixels from consideration by setting their conf to 1 (high confidence so they won't be selected)
                    conf = conf.masked_fill(~valid_mask, 1.0)
                    # Sort confidences of valid pixels in ascending order (low confidence = hard examples first)
                    conf_flat = conf[valid_mask]
                    if conf_flat.numel() > 0:
                        sort_conf, _ = torch.sort(conf_flat)  # ascending sort
                        # Determine threshold to keep at least min_kept 
                        # Get the confidence value at the index self.min_kept*batch size
                        batch_kept = self.min_kept * B
                        last_index = conf_flat.numel() - 1
                        thresh_index = min(batch_kept, last_index) #ensure that it does not exceed valid max
                        min_conf = sort_conf[thresh_index]      # minimum confidence among the top-K hardest
                        
                        # Guarantees at least min_kept pixel to be selected
                        # if self.thresh can not select enough pixel(topK), increase threshold to the index of topK
                        effective_thresh = max(min_conf, self.thresh)  # final threshold
                    else:
                        effective_thresh = self.thresh  # if no valid pixel, use given thresh as fallback
                    # Select pixels with confidence below the threshold as hard examples
                    seg_weight = (conf < effective_thresh) & valid_mask  # bool mask of hard pixels:contentReference[oaicite:12]{index=12}
                    seg_weight = seg_weight.float()  # convert to float mask (1.0 for selected, 0.0 otherwise)
            else:
                # Loss-based hard example selection (take top-K losses)
                # Compute per-pixel BCE loss without reduction
                bce_loss_map = F.binary_cross_entropy_with_logits(pred_logits, target_mask.float(), reduction='none')
                # Mask out ignore pixels by setting their loss to 0 (so they won't be selected)
                if self.ignore_index is not None:
                    bce_loss_map = bce_loss_map * valid_mask.float()
                # Flatten the losses of valid pixels and sort in descending order (hardest first)
                if self.reduction == 'none':
                    seg_weight = self._per_image_topk_mask(bce_loss_map, valid_mask)
                else:
                    loss_flat = bce_loss_map[valid_mask]
                    if loss_flat.numel() > 0:
                        sorted_loss, sorted_idx = torch.sort(loss_flat, descending=True)
                        batch_kept = self.min_kept * B
                        # Create a mask for selected top losses
                        selected_idx = sorted_idx[:batch_kept]  # indices of top `batch_kept` 
                        seg_weight = torch.zeros_like(loss_flat, dtype=torch.float)
                        seg_weight[selected_idx] = 1.0  # mark selected hard examples
                        # Map the selection back to the original spatial shape
                        seg_weight_full = torch.zeros_like(bce_loss_map, dtype=torch.float)  # (B, H, W)
                        seg_weight_full[valid_mask] = seg_weight
                        seg_weight = seg_weight_full
                    else:
                        # If no valid pixels (unlikely unless all are ignore), use zeros
                        seg_weight = torch.zeros_like(target_mask, dtype=torch.float)
            # `seg_weight` is a float mask (B,H,W) with 1 for selected pixels and 0 for others.
        
        # Recompute per-pixel BCE with gradients as bce_loss_map (B,H,W).
        bce_loss_map = F.binary_cross_entropy_with_logits(pred_logits, target_mask.float(), reduction='none')
        # Multiply by seg_weight to zero out easy/ignored pixels: bce_loss_weighted.
        bce_loss_weighted = bce_loss_map * seg_weight
        # Compute mean loss over the selected pixels (avoid division by zero)
        if self.reduction == 'none':
            # per-image BCE = sum(selected losses) / count(selected) per image
            sel_counts = seg_weight.sum(dim=(1, 2))                              # (B,)
            bce_per_img = torch.where(
                sel_counts > 0,
                bce_loss_weighted.sum(dim=(1, 2)) / (sel_counts + 1e-12),
                torch.zeros_like(sel_counts, dtype=bce_loss_weighted.dtype, device=bce_loss_weighted.device)
            )                                                                     # (B,)

            # Dice per image; ensure shape (B,)
            dice_per_img = self.dice_loss_fn(pred_logits, target_mask)            # (B,) or (B,1)
            if dice_per_img.dim() == 2 and dice_per_img.size(1) == 1:
                dice_per_img = dice_per_img.squeeze(1)

            total_loss = self.dice_weight * dice_per_img + self.bce_weight * bce_per_img  # (B,)
            return total_loss

        elif self.reduction == 'sum':
            # compute per-image, then sum across images
            sel_counts = seg_weight.sum(dim=(1, 2))                              # (B,)
            bce_per_img = torch.where(
                sel_counts > 0,
                bce_loss_weighted.sum(dim=(1, 2)) / (sel_counts + 1e-12),
                torch.zeros_like(sel_counts, dtype=bce_loss_weighted.dtype, device=bce_loss_weighted.device)
            )                                                                     # (B,)
            dice_per_img = BinaryDiceLoss(reduction='none')(pred_logits, target_mask)  # (B,) or (B,1)
            if dice_per_img.dim() == 2 and dice_per_img.size(1) == 1:
                dice_per_img = dice_per_img.squeeze(1)

            total_per_img = self.dice_weight * dice_per_img + self.bce_weight * bce_per_img  # (B,)
            return total_per_img.sum() # scalar

        else:  # 'mean' 
            if torch.any(seg_weight > 0):
                bce_loss = bce_loss_weighted.sum() / seg_weight.sum() # scalar
            else:
                bce_loss = torch.tensor(0.0, device=pred_logits.device)

            dice_loss = self.dice_loss_fn(pred_logits, target_mask)  # scalar
            total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss # scalar
            return total_loss

class TopoGuide():
    def __init__(self, filtrationType = 'VR', distanceMetric='wasserstein', 
                 epsilon=50, reduction='sum'):

        self.filtrationType = filtrationType
        self.distanceMetric= distanceMetric
        self.reduction=reduction
        self.epsilon = epsilon #  # for geometric and diffusion graph construction

    # function with which distance between persistence is calculated
    def distanceCalculate(self, gold, pred):
        H0gold, H1gold = gold
        H0pred, H1pred = pred
        if self.distanceMetric=='wasserstein':
            # order=1  the Earth Mover’s Distance, where costs are summed linearly 
            # internal_p specifies how to compute the distance between two such points (birth, death) or between a point and the diagonal in R2, p=2 euclidean distance
            H0dist = gd.wasserstein.wasserstein_distance(H0pred, H0gold, order=1, internal_p=2)
            H1dist = gd.wasserstein.wasserstein_distance(H1pred, H1gold, order=1, internal_p=2)
        
        if self.reduction=='mean':
            return (H1dist+H0dist)/2
        elif self.reduction=='sum':
            return H1dist+H0dist
        elif self.reduction=='H1':
            return H1dist

            
    def extractPoints(self, BinarySegMap):
        # get dot predictions (centers of connected components)
        contours, hierarchy = cv2.findContours(BinarySegMap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        e_coord_y = []
        e_coord_x = []
        for idx in range(len(contours)):
            #print('idx=',idx)
            contour_i = contours[idx]
            M = cv2.moments(contour_i)
            #print(M)
            if(M['m00'] == 0):
                continue;
            cx = round(M['m10'] / M['m00'])
            cy = round(M['m01'] / M['m00'])
            e_coord_y.append(cy)
            e_coord_x.append(cx)

        e_coord = np.zeros((len(e_coord_y),2), dtype=int)  # force integer dtype)
        e_coord[:,0] = np.array(e_coord_x)
        e_coord[:,1] = np.array(e_coord_y)
        return e_coord


    
    def VietorisRipsFiltration(self, coordinates):
        if len(coordinates) == 0:
            return [np.empty((0,2)), np.empty((0,2))]
        skeleton = gd.RipsComplex(points = coordinates, max_edge_length = self.epsilon)
        # max_dimension=2 refers to simplex dimension
        # This means:
        #     The filtration will include:
        #         0-simplices (points)
        #         1-simplices (edges)
        #         2-simplices (triangles) — which are required to "fill in" loops and compute H1 deaths
        Rips_simplex_tree = skeleton.create_simplex_tree(max_dimension = 2)
        Rips_simplex_tree.persistence() 
        H1 = Rips_simplex_tree.persistence_intervals_in_dimension(1)
        H0 = Rips_simplex_tree.persistence_intervals_in_dimension(0)
        H0 = np.nan_to_num(H0, posinf=self.epsilon, neginf=0)
        H1 = np.nan_to_num(H1, posinf=self.epsilon, neginf=0)
        return [H0, H1]
        
    
    def __call__(self, gt_dot, segMapPred):

        #extract cell coorditanes for label
        gt_y, gt_x = np.where(gt_dot!=0)
        coordinates_gold = np.zeros((len(gt_y),2))
        coordinates_gold[:,0] = np.array(gt_x)
        coordinates_gold[:,1] = np.array(gt_y)
        #extract cell coorditanes for prediction
        coordinates_pred = self.extractPoints(segMapPred)
                
        # defaults: empty PDs on both sides
        homologyGold = [np.empty((0, 2)), np.empty((0, 2))]
        homologyPred = [np.empty((0, 2)), np.empty((0, 2))]
        
        # ---- GOLD ----
        n_g = len(coordinates_gold)
        if n_g >= 1:
            if self.filtrationType == 'VR':
                homologyGold = self.VietorisRipsFiltration(coordinates_gold)
            else:
                print('Currently only VR filtration is implemented for GT. Please set filtrationType to VR.')

        # ---- PRED ----
        n_p = len(coordinates_pred)
        if n_p >= 1:
            if self.filtrationType == 'VR':
                homologyPred = self.VietorisRipsFiltration(coordinates_pred)
            else:
                print('Currently only VR filtration is implemented for prediction. Please set filtrationType to VR.')

        loss = self.distanceCalculate(homologyGold, homologyPred)

        return loss
    
    
class TGHEM(nn.Module):
    """
    Main Loss Class: Combines Segmentation Loss with Topological Reweighting.
    """
    def __init__(self, lambda_pers=0.5, ohem_thresh=0.7, ohem_min_kept=15000, 
                 topo_epsilon=50, device='cuda'):
        super().__init__()
        self.lambda_pers = lambda_pers
        
        # Initialize sub-modules
        self.seg_loss = SegmentationLoss(thresh=ohem_thresh, min_kept=ohem_min_kept, reduction='none')
        self.topo_guide = TopoGuide(epsilon=topo_epsilon, reduction='sum')
        
    def forward(self, pred_logits, target_tuple):
        """
        Args:
            pred_logits: (B, 1, H, W)
            target_tuple: (gt_mask, gt_dots) 
                          gt_mask for pixel loss, gt_dots for topology
        """
        gt_mask, gt_dots = target_tuple
        
        # 1. Compute Pixel-wise Segmentation Loss (Per Image)
        # Returns (B,) tensor
        loss_seg_per_img = self.seg_loss(pred_logits, gt_mask)
        
        # 2. Compute Topological Weights (No Gradient Flow)
        batch_size = gt_mask.shape[0]
        topo_discrepancy = []
        
        # Detach and binarize for topology calculation
        with torch.no_grad():
            prob = torch.sigmoid(pred_logits.squeeze(1))  # (B, H, W)
            pred_bin = (prob >= 0.5).cpu().numpy().astype(np.uint8)
            gt_dots_np = gt_dots.cpu().numpy().astype(np.uint8)
            
            for b in range(batch_size):
                t_dis = self.topo_guide(gt_dots_np[b], pred_bin[b])
                topo_discrepancy.append(t_dis)
        
        # Convert list to tensor
        topo_discrepancy = torch.tensor(topo_discrepancy, device=pred_logits.device, dtype=pred_logits.dtype)
        
        # 3. Apply Reweighting
        # Weight = 1 + lambda * topological_discrepancy
        weights = 1.0 + (self.lambda_pers * topo_discrepancy)
        
        # Weighted mean over batch
        loss = (weights * loss_seg_per_img).sum() / batch_size
        
        return loss
    
def main():
    # Initialize
    criterion = TGHEM(
        lambda_pers=0.01, 
        ohem_thresh=0.7,
        ohem_min_kept=15000, 
        topo_epsilon=50
    ).cuda()

    # In training loop
    # pred: (B, 1, H, W) logits from model
    # gt_map: (B, H, W) binary mask
    # gt_dot: (B, H, W) binary mask of centroids/dots
    loss = criterion(pred, (gt_map, gt_dot))
    loss.backward()