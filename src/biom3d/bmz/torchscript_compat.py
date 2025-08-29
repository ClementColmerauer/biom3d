"""
I'm not 100% sure that this code would work with 2d images.

Thing that are not torchscript compatible:
- Numpy
- Everything that is dynamic
- Dynamic typing, variable type are either infered or type hinted (if infered not possible)
"""

import torch.nn.functional as F
from typing import List,Tuple
import torch

#Based on torchio gridsampler
class GridSampler:
    """
    Torchscript compatible adaptation of `torchio.data.GridSampler`, pad and cut an image in patches.
    """
    def __init__(self,
                img: torch.Tensor, 
                patch_size: torch.Tensor, 
                patch_overlap: torch.Tensor):
        """
        Torchscript compatible adaptation of `torchio.data.GridSampler`, pad and cut an image in patches. Store patches in `self.patches`, location in `self.patches_location` and new image size in `self.padded_size`.
        
        Parameters
        ----------
        img : torch.Tensor
            The image to sample.
        patch_size : torch.Tensor
            Size of the patch used during training.
        patch_overlap: torch.Tensor
            Overlapping of patches. 
        """
        self._img = img
        self._patch_overlap = patch_overlap
        self._patch_size = patch_size
        self._img = self._pad()
        self.patches_location = self._compute_location()
        self.patches = self._generate_patch()
        self.padded_shape = self._img.shape

    def _pad(self)->torch.Tensor:
        """
        Apply symetric pad so patch stay in the image bound
        
        Returns
        ----------
        torch.Tensor
            Padded version of `self._img`
        """
        border = self._patch_overlap // 2
        padding:List[int] = [int(border[2].item()), int(border[2].item()),
                    int(border[1].item()), int(border[1].item()),
                    int(border[0].item()), int(border[0].item())]

        padded_img = F.pad(self._img, padding, mode='constant', value=0.0)
        
        return padded_img

    def _lexsort_tensor(self,data: torch.Tensor) -> List[Tuple[int, int, int, int, int, int]]:
        """
        Torchscript friendly (and adaptated to our format) version of lexsort then toList().

        Parameters
        ----------
        data: torch.Tensor
            The data to sort

        Returns
        -------
        list of tuple of int
            Sorted version of `data` as a List of tuple `(ystart,xstart,zstart,yend,xend,zend)`
        """
        indices = torch.arange(data.size(0))
        num_cols = data.size(1)
        for col in range(num_cols-1,-1,-1):
            keys = data[:, col]
            _, new_order = torch.sort(keys[indices], stable=True)
            indices = indices[new_order]     
        sorted_data = data[indices]
        output: List[Tuple[int, int, int, int, int, int]] = []
        for i in range(sorted_data.size(0)):
            row = sorted_data[i]
            output.append((
                int(row[0]), int(row[1]), int(row[2]),
                int(row[3]), int(row[4]), int(row[5])
            ))
        return output
    
    def _compute_location(self) -> List[Tuple[int, int, int, int, int, int]]:
        """
        Compute location of patches and return them in a deterministicly sorted list
          
        Returns
        -------
        list of tuple of int
            Patches location as a List of tuple `(ystart,xstart,zstart,yend,xend,zend)`
        """
        indexes:List[List[int]] = []
        size = self._img.shape[-3:]

        #Computation of starting indexes
        for i in range(len(size)):
            im_size_dim = size[i]
            patch_size_dim:int = int(self._patch_size[i].item())
            patch_overlap_dim:int = int(self._patch_overlap[i].item())      
            end:int = im_size_dim + 1 - patch_size_dim
            step:int = patch_size_dim - patch_overlap_dim
            indices_dim:List[int] = list(range(int(0),end,step))
            if indices_dim[-1] != im_size_dim - patch_size_dim: indices_dim.append(im_size_dim - patch_size_dim)
            indexes.append(indices_dim)

        #Computation of patch grid
        grids = []
        for dim in indexes:
            grids.append(torch.tensor(dim, dtype=torch.int32))
        grid = torch.meshgrid(grids, indexing="xy")
        grid = [m.flatten() for m in grid]

        # Grid to list
        indexes_start = torch.stack(grid,dim=1)
        indexes_start = torch.unique(indexes_start,dim=0)
        index_end = indexes_start + self._patch_size
        locations = torch.hstack((indexes_start,index_end))

        #Lexsort for determinism
        return self._lexsort_tensor(locations)
    
    def _crop(self,img:torch.Tensor,index_start:Tuple[int,int,int]):
        """
        Generate the patch at given location by cropping an image.

        Parameters
        ----------
        img: torch.Tensor
            The image to crop
        index_start : Tuple[int,int,int])
            Stating indexes of the patch.
          
        Returns
        -------
        tuple of int
            The patch
        """
        d0, h0, w0 = index_start
        #Using patch size as security
        pd = self._patch_size[0].item()
        ph = self._patch_size[1].item()
        pw = self._patch_size[2].item()
        d1 = d0 + pd
        h1 = h0 + ph
        w1 = w0 + pw
        patch = img[:, d0:d1, h0:h1, w0:w1]
        return patch
    
    def _generate_patch(self)->List[torch.Tensor]:      
        """
        Generate the patches with `self.patches_location`.
          
        Returns
        ----------
        list of torch.Tensor
            A list of patch in the same order as `self.patches_location`
        """      
        patches = []
        for l in self.patches_location:
            index_ini = l[:3]
            patches.append(self._crop(self._img,index_ini))
        return patches

#Based on torchio gridaggregator
class GridAggregator():
    """
    Torchscript compatible adaptation of `torchio.data.GridAggregator`, aggregate a list of prediction and another list of those predicttion location to a logit stored in self.logit
    """
    def __init__(self,
                 patch_preds:List[torch.Tensor],
                 patch_loc_preds:List[Tuple[int,int,int,int,int,int]], 
                 output_shape:torch.Tensor, 
                 patch_size:torch.Tensor, 
                 padded_size:List[int]):
        """
        Torchscript compatible adaptation of `torchio.data.GridAggregator`, aggregate a list of prediction and another list of those predicttion location to a logit stored in self.logit
        
        Parameters
        ----------
        patch_preds : List[torch.Tensor]
            The list of predicted patch (not batch).
        patch_loc_preds : List[Tuple[int,int,int,int,int,int]]
            The list of predicted patch location in the same order.
        output_shape : torch.Tensor
            The shape of the image before the Sampling, Tensor of size 3 or 4.
        patch_size : torch.Tensor
            Size of the patch used during training.
        padded_size : List[int]
            Size of the image after sampling
        """
        _,D,H,W = padded_size
        C,_,_,_ = patch_preds[0].shape

        #Case where it has only spatial dimension 
        if len(output_shape) == 3:
            Do = int(output_shape[0].item())
            Ho = int(output_shape[1].item())
            Wo = int(output_shape[2].item())
            output_shape = torch.tensor((C,Do,Ho,Wo))

        self._patch_size = patch_size   
        self._hann = self._get_hann_window()
        self.logit = torch.zeros((C, D, H, W), dtype=patch_preds[0].dtype)
        self._avgmask_tensor = torch.zeros((C, D, H, W), dtype=patch_preds[0].dtype)
        for i in range(len(patch_preds)):
            self._add_patch_hann(patch_preds[i],patch_loc_preds[i])
        
        self.logit = self._finalize_output()
        padded_size_tensor = torch.tensor(padded_size)
        if len(padded_size_tensor) == 3:
            padded_size_tensor.unsqueeze(dim=0)
            padded_size_tensor[0]= C
        self.logit = self._crop(padded_size_tensor-output_shape)  

    def _get_hann_window(self)->torch.Tensor:
        """
        Get the hann window needed for aggregation.

        Returns 
        -------
        torch.Tensor
            A 3d hann window tensor
        """
        hann_window = torch.ones(1)
        for i in range(3):
            size = self._patch_size[i]
            window_1d = torch.hann_window(size + 2, periodic=False)[1:-1]
            shape = [1, 1, 1]
            shape[i] = size.item()
            shape_tuple = (shape[0],shape[1],shape[2])
            window_1d = window_1d.view(shape_tuple)
            hann_window = hann_window * window_1d
        return hann_window
    
    def _add_patch_hann(self,
            patch: torch.Tensor,
            location: Tuple[int,int,int,int,int,int],
        ) -> None:
        """    
        Add the patch to the logit, ponderating with an hann window

        Parameters 
        ----------
        patch : torch.Tensor
            The patch to had
        location: Tuple[int,int,int,int,int,int]
            The location of the patch

        Returns
        -------
        None
        """
        weighted_patch = patch * self._hann
        i0 = int(location[0])
        j0 = int(location[1])
        k0 = int(location[2])
        i1 = int(location[3])
        j1 = int(location[4])
        k1 = int(location[5])
        self.logit[:, i0:i1, j0:j1, k0:k1] += weighted_patch
        self._avgmask_tensor[:, i0:i1, j0:j1, k0:k1] += self._hann

    def _finalize_output(self) -> torch.Tensor:
        """      
        Compute the mean of the logit by the average mask tensor

        Returns 
        ----------
        torch.Tensor
            The logit, still in padded size
        """
        return torch.true_divide(self.logit, self._avgmask_tensor)
    
    def _crop(self,border: torch.Tensor) -> torch.Tensor:
        """
        Crop symetrically the logit by the `border`

        Parameters 
        ----------
        border : torch.Tensor
            A tensor of size 3 that describe by how much the padding must be done it the 3 spatial dimension.

        Returns 
        ----------
        torch.Tensor
            The logit
        """
        border = border[-3:] // 2
        border_list: List[int]= border.tolist()
        d_b, h_b, w_b = border_list
        return self.logit[:, 
                    d_b:-d_b if d_b > 0 else None,
                    h_b:-h_b if h_b > 0 else None,
                    w_b:-w_b if w_b > 0 else None]

class SizeFilter:
    @staticmethod
    def _dist_vec(v1: torch.Tensor, v2: torch.Tensor) -> float:
        """
        Euclidean distance between two 1D torch tensors.

        Parameters
        ----------
        v1 : torch.Tensor
            Vector 1
        v2 : torch.Tensor
            Vector 2

        Returns
        -------
        torch.Tensor
            Euclidean distance between v1 and v2.
        """
        v = v2 - v1
        return torch.sqrt(torch.sum(v * v)).item()
    
    @staticmethod
    def _center(labels:torch.Tensor, idx:int)->torch.Tensor:
        """
        Compute the barycenter of pixels belonging to a specific label.

        Parameters
        ----------
        labels : torch.Tensor
            Label image array where each pixel has an integer label.
        idx : int
            Label index for which to compute the barycenter.

        Returns
        -------
        torch.Tensor
            Coordinates of the barycenter as a 1D array (e.g. [y, x] or [z, y, x] depending on dimensions).
            If no pixels with the given label are found, returns an empty array.
        """
        
        return torch.mean((torch.argwhere(labels.to(torch.float32) == float(idx))).to(torch.float32), dim=0)
    
    @staticmethod
    def _otsu_thresholding(im:torch.Tensor)->torch.Tensor:
        """
        Compute the optimal threshold for an image using Otsu's method.

        This function searches for the threshold value that minimizes the
        weighted within-class variance of the thresholded image.

        Parameters
        ----------
        im : torch.Tensor
            Grayscale input image as a 2D numpy array.

        Returns
        -------
        torch.Tensor
            Optimal threshold value computed using Otsu's method.
        """
        threshold_range = torch.linspace(im.min(), im.max()+1, steps=255)
        criterias = torch.tensor([SizeFilter._compute_otsu_criteria(im, th) for th in threshold_range])
        best_th = threshold_range[torch.argmin(criterias,dim=None)]
        return best_th
    
    @staticmethod
    def _compute_otsu_criteria(im, th):
        """
        Compute the Otsu criteria value for a given threshold on the image.

        This function implements the core step of Otsu's method, which evaluates
        the within-class variance weighted by class probabilities for a specific threshold.
        The goal is to find the threshold minimizing this weighted variance.
        Found here: https://en.wikipedia.org/wiki/Otsu%27s_method.

        Parameters
        ----------
        im : torch.Tensor
            Grayscale input image as a 2D numpy array.
        th : torch.Tensor
            Threshold value to evaluate.

        Returns
        -------
        torch.Tensor
            Weighted sum of variances for the two classes separated by the threshold.
            Returns `np.inf` if one class is empty (to ignore this threshold).
        """
        # create the thresholded image
        thresholded_im = torch.zeros(im.shape)
        thresholded_im[im >= th] = 1

        # compute weights
        nb_pixels = im.numel()
        nb_pixels1 = torch.count_nonzero(thresholded_im,dim=None)
        weight1 = nb_pixels1 / nb_pixels
        weight0 = 1 - weight1

        # if one of the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
        # in the search for the best threshold
        if weight1 == 0 or weight0 == 0:
            return torch.tensor(float("inf"))

        # find all pixels belonging to each class
        val_pixels1 = im[thresholded_im == 1].to(torch.float32)
        val_pixels0 = im[thresholded_im == 0].to(torch.float32)

        # compute variance of these classes
        var1 = torch.var(val_pixels1, dim=None) if val_pixels1.numel() > 1 else torch.tensor(0.0)
        var0 = torch.var(val_pixels0, dim=None) if val_pixels0.numel() > 1 else torch.tensor(0.0)

        return weight0 * var0 + weight1 * var1
    
    @staticmethod
    def _volumes(labels:torch.Tensor)->torch.Tensor:
        """
        Compute the volume (pixel or voxel count) of each label in the label image.

        Parameters
        ----------
        labels : torch.Tensor
            Label image array where each pixel has an integer label.

        Returns
        -------
        torch.Tensor
            Array of counts of pixels per label, sorted by label index ascending.
        """
        return torch.unique(labels, return_counts=True,dim=None)[1]
    
    @staticmethod
    def _get_neighbors(z: int, y: int, x: int,D:int,H:int,W:int) -> List[Tuple[int, int, int]]:
        """
        Get the voxels with the same value, neighbors of a given location.

        Parameters
        ----------
        z: int
            Z coordinate
        y: int
            Y coordinate
        x: int
            X coordinate
        D: int
            Max Z coordinate
        H: int
            Max Y coordinate
        W: int
            Max X coordinate
        
        Returns
        -------
        list of tuple (z,y,x)
        """
        offsets = [-1, 0, 1]
        neighbors = torch.jit.annotate(List[Tuple[int, int, int]], [])
        for dz in offsets:
            for dy in offsets:
                for dx in offsets:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    nz = z + dz
                    ny = y + dy
                    nx = x + dx
                    if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                        neighbors.append((nz, ny, nx))
        return neighbors

    @staticmethod
    def _label(msk:torch.Tensor)->Tuple[torch.Tensor,int]:
        """
        Label connected components in a 3D binary mask using iterative flood fill.

        Parameters
        ----------
        msk : torch.Tensor
            A 3D binary tensor of shape (D, H, W), where `1` indicates foreground
            voxels and `0` indicates background.

        Returns
        -------
        labels : torch.Tensor
            A 3D tensor of the same shape as `msk` where each connected component
            is assigned a unique integer label (starting from 1). Background voxels remain 0.

        num_labels : int
            The number of connected components found in the input mask.
        """
        D, H, W = msk.shape
        labels = torch.zeros((D, H, W), dtype=torch.int32)
        current_label = 1

        

        for z in range(D):
            for y in range(H):
                for x in range(W):
                    if msk[z, y, x] == 0 or labels[z, y, x] != 0:
                        continue
                    stack = torch.jit.annotate(List[Tuple[int, int, int]], [])
                    stack.append((z, y, x))

                    while len(stack) > 0:
                        cz, cy, cx = stack.pop()
                        if labels[cz, cy, cx] != 0:
                            continue
                        labels[cz, cy, cx] = current_label

                        neighbors = SizeFilter._get_neighbors(cz, cy, cx,D,H,W)
                        for nz, ny, nx in neighbors:
                            if msk[nz, ny, nx] == 1 and labels[nz, ny, nx] == 0:
                                stack.append((nz, ny, nx))

                    current_label += 1

        return labels,current_label -1
    
    @staticmethod
    def _keep_center_only(msk:torch.Tensor)->torch.Tensor:        
        """
        Keep only the connected component in the mask that is closest to the image center.

        Parameters
        ----------
        msk : torch.Tensor
            Binary mask (2D or 3D) where connected components are to be analyzed.

        Returns
        -------
        torch.Tensor
            Mask with only the connected component closest to the center.
            The returned mask has the same dtype as input, with values 0 or 255.
        """
        labels, num = SizeFilter._label(msk)
        close_idx = SizeFilter._closest(labels,num)
        return (labels==close_idx).astype(msk.dtype)*255
    
    @staticmethod
    def _closest(labels:torch.Tensor, num:int)->torch.Tensor:
        """
        Find the label index of the object closest to the center of the image.

        The function computes the barycenter of all objects (labels 1 to num),
        then returns the label of the object whose barycenter is closest to the image center.

        Parameters
        ----------
        labels : torch.Tensor
            Label image array where each pixel has an integer label.
        num : int
            Number of labels (excluding background) to consider.

        Returns
        -------
        int
            The label index (1-based) of the object closest to the image center.
            Returns 1 if no objects are found.
        """
        labels_center = torch.tensor(labels.shape, dtype=torch.float32) / 2
        centers = [SizeFilter._center(labels,idx+1) for idx in range(num)]
        dist = torch.tensor([SizeFilter._dist_vec(labels_center,c) for c in centers])
        # bug fix, return 1 if dist is empty:
        if len(dist)==0:
            return torch.tensor(1)
        else:
            return torch.argmin(dist,dim=None)+1
    
    @staticmethod
    def keep_big_volumes(msk:torch.Tensor, thres_rate:float=0.3)->torch.Tensor:
        """
        Return a mask keeping only the largest connected components based on a volume threshold.

        The threshold is computed as: min_volume = thres_rate * otsu_thresholding(volumes)
        where `volumes` are the sizes of all connected components (excluding background),
        and `otsu_thresholding` finds an adaptive threshold on the volumes distribution.

        Parameters
        ----------
        msk : torch.Tensor
            Input binary mask.
        thres_rate : float, default=0.3
            Multiplier for the threshold on volumes.

        Returns
        -------
        torch.Tensor
            Mask with only the connected components whose volume is greater than the threshold.
            Background remains zero.
        """
        # transform image to label
        labels, num = SizeFilter._label(msk)

        # if empty or single volume, return msk
        if num <= 1:
            return msk

        # compute the volume
        unq_labels,vol = torch.unique(labels, return_counts=True,dim=None)

        # remove bg
        unq_labels = unq_labels[1:]
        vol = vol[1:]

        # compute the expected volume
        # expected_vol = np.sum(np.square(vol))/np.sum(vol)
        # min_vol = expected_vol * thres_rate
        min_vol = thres_rate*SizeFilter._otsu_thresholding(vol)

        # keep only the labels for which the volume is big enough
        unq_labels = unq_labels[vol > min_vol]

        # compile the selected volumes into 1 image
        s = (labels==unq_labels[0])
        for i in range(1,len(unq_labels)):
            s += (labels==unq_labels[i])

        return s
    
    @staticmethod
    def keep_biggest_volume_centered(msk:torch.Tensor)->torch.Tensor:
        """
        Return a mask with only the connected component closest to the image center, provided its volume is not too small compared to the largest connected component. Otherwise, return the largest connected component.

        "Too small" means its volume is less than half of the largest component.

        The returned mask intensities are either 0 or `msk.max()`.

        Parameters
        ----------
        msk : torch.Tensor
            Input binary mask.

        Returns
        -------
        torch.Tensor
            Mask with only one connected component kept.
        """
        labels, num = SizeFilter._label(msk)
        if num <= 1: # if only one volume, no need to remove something
            return msk
        close_idx = SizeFilter._closest(labels,num)
        vol = SizeFilter._volumes(labels)
        relative_vol = torch.tensor([vol[close_idx]/vol[idx] for idx in range(1,len(vol))])
        # bug fix, empty prediction (it should not happen)
        if len(relative_vol)==0:
            return msk
        min_rel_vol = torch.min(relative_vol)
        if min_rel_vol < 0.5:
            close_idx = torch.argmin(relative_vol,dim=None)+1
        return (labels==close_idx).to(msk.dtype)*msk.max()

class PostProcessing:
    """Class that encapsulate different torchscript compatible post processing."""
    @staticmethod
    def seg_postprocessing(logit:torch.Tensor,
                           use_softmax:bool,
                           force_softmax : bool, 
                           return_logit : bool, 
                           keep_big_only:bool,
                           keep_biggest_only:bool)->torch.Tensor:
        """
        Post-process the logit (model output) to obtain the final segmentation mask. Can optionally remove some noise. 

        Recommanded to be used after biom3d.predictors.seg_predict_patch_2.
    
        Parameters
        ----------
        logit : torch.Tensor
            The raw model output.
        use_softmax: bool
            Whether softmax was used at training (default behaviour)
        force_softmax: bool
            If training done without softmax, force the postprocessing to use it to create the mask
        keep_big_only: bool
            Postprocessing to remove noise by keeping the bigger object, defined by an Otsu thresoldhing (not sure if work for 2D)
        keep_biggest_only: bool
            Postprocessing to remove noise by keeping the biggest centered object (not sure if work for 2D)
        return_logit: bool
            Whether to skip post processing or not.

        Returns
        -------
        torch.tensor
            The post-processed segmentation mask or logit.
        """
        num_classes:int = logit.shape[0]

        if return_logit: 
            return logit

        if use_softmax:
            out = (logit.softmax(dim=0).argmax(dim=0)).int() 
        elif force_softmax:
            # if the training has been done with a sigmoid activation and we want to export a softmax
            # it is possible to use `force_softmax` argument
            sigmoid = (logit.sigmoid()>0.5).int()
            softmax = (logit.softmax(dim=0).argmax(dim=0)).int()+1
            cond = sigmoid.max(dim=0).values
            out = torch.where(cond>0, softmax, 0)  
        else:
            out = (logit.sigmoid()>0.5).int()
                  
        if keep_big_only and keep_biggest_only:
            print("[Warning] Incompatible options 'keep_big_only' and 'keep_biggest_only' have both been set to True. Please deactivate one! We consider here only 'keep_biggest_only'.")
        if keep_biggest_only or keep_big_only:
            if use_softmax: # then one-hot encode the net output
                out = out.to(torch.long)
                out = F.one_hot(out, num_classes=num_classes).to(torch.int)
                if len(out)==3: out = out.permute(2,0,1)
                elif len(out)==4: out = out.permute(3,0,1,2)
                else: raise RuntimeError()

            if len(out.shape)==3:
                out = SizeFilter.keep_big_volumes(out) if keep_big_only else SizeFilter.keep_biggest_volume_centered(out)
            elif len(out.shape)==4:
                tmp = []
                for i in range(out.shape[0]):
                    tmp += [SizeFilter.keep_big_volumes(out[i]) if keep_big_only else SizeFilter.keep_biggest_volume_centered(out[i])]
                out = torch.stack(tmp)
                
            if use_softmax: # set back to non-one-hot encoded
                out = out.argmax(0)

        out = out.to(torch.uint8)    
        return out

class ModelExport(torch.nn.Module):
    """Class that encapsulate the prediction pipeline."""
    def __init__(self,
                model,
                original_shape:torch.Size,
                axes_order:str,
                patch_size:list,
                num_workers:int = 4,
                device = 'cpu',
                tta:bool= False, 
                batch_size=2,
                use_softmax:bool=True,
                force_softmax:bool=False,
                keep_big_only:bool=False,
                keep_biggest_only:bool=False,
                return_logit:bool=False):
        """"
        Initialize the model.

        Parameters
        ----------
        model: torch.nn.Module
            A torchsript compatible module (do not type hint it torchscript doesn't understant dynamic class)
        original_shape: torch.Size
            Shape of the image, it was used for resampling but not implemented in torchscript.
        axes_order: str
            Not used
        patch_size: list of int of size 3 (or 2 for 2d)
            Size of a patch
        num_worker: int, default=4
            Number of process used, if 1, no subprocess is started
        device: strndefault=cpu
            Device used for compilation, must be same as destination, only 'cpu','cuda' and 'mps' work.
        tta: bool, default=False
            Whether data augmentation is used during prediction (longer but more precise)
        batch_size: int
            Size of a single batch
        use_softmax: bool, default=True
            Whether softmax was used at training (default behaviour)
        force_softmax: bool, default=False
            If training done without softmax, force the postprocessing to use it to create the mask
        keep_big_only: bool, default=False
            Postprocessing to remove noise by keeping the bigger object, defined by an Otsu thresoldhing (not sure if work for 2D)
        keep_biggest_only: bool, default=False
            Postprocessing to remove noise by keeping the biggest centered object (not sure if work for 2D)
        return_logit: bool, default=False
            Whether to skip post processing or not.
        """
        super().__init__()
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.tta = tta
        self.axes_order = axes_order
        self.original_shape = original_shape
        self.use_softmax=use_softmax
        self.force_softmax=force_softmax
        self.keep_big_only=keep_big_only
        self.keep_biggest_only=keep_biggest_only
        self.return_logit=return_logit

        self.model = model.to(self.device).eval()
        self.model.to(self.device)

    def process_batch(self,
                      X:torch.Tensor,
                      batch_locations:List[Tuple[int,int,int,int,int,int]],
                      pred_aggr:List[torch.Tensor],
                      patch_loc_preds:List[Tuple[int,int,int,int,int,int]]
                      )->None:
        """
        Make prediction to a batch and add it to a prediction list.
        
        Parameters
        ----------
        X: torch.Tensor
            The batch to process: NCZYX
        batch_location : list of location
            The locations of the different elements of the batch, is added to prediction location
        pred_aggr: list of torch.Tensor
            List of predictions
        patch_loc_pred: list of location
            List of the prediction location

        Returns
        -------
        None
        """
        with torch.no_grad():
            if self.device == 'cuda':
                X = X.cuda()
            elif self.device == 'mps':
                X = X.to('mps')
            
            # Contrarly to seg_patch_2 we don't use autocast as torchscript lose even more precision
            # Hence, exported model will be slower but more accurate on GPU
            if self.tta: # test time augmentation: flip around each axis
                pred=self.model(X).cpu()
                
                # flipping tta
                dims = [[2],[3],[4],[3,2],[4,2],[4,3],[4,3,2]]
                for d in dims :
                    X_flip = torch.flip(X,dims=d)

                    pred_flip = self.model(X_flip)
                    pred += torch.flip(pred_flip, dims=d).cpu()
                    
                    
                
                pred = pred/(len(dims)+1)
            else:
                pred=self.model(X).cpu()
            for j in range(len(pred)):
                pred_aggr.append(pred[j])
                patch_loc_preds.append(batch_locations[j])

    def forward(self,img:torch.Tensor)->torch.Tensor:
        """
        Forward pass of the model.

        1. Sample the image
        2. Create batches
        3. Delegate batches prediction to model
        4. Aggregate prediction
        5. Delegate post processing

        Paramters
        ---------
        img: torch.Tensor
            An image to predict
        
        Returns
        -------
        logit: torch.Tensor
            Post processed output of the model
        """
        overlap = 0.5
        patch_size = torch.tensor(self.patch_size,dtype=torch.int)
        original_shape = torch.tensor(self.original_shape)
        patch_overlap = torch.maximum(torch.mul(patch_size,overlap),torch.sub(patch_size, torch.tensor(img.shape[-3:])))
        patch_overlap = torch.div(torch.ceil(torch.mul(patch_overlap,overlap)),overlap).to(torch.int)
        patch_overlap = torch.minimum(patch_overlap,torch.sub(patch_size,1))
        pred_aggr =  torch.jit.annotate(List[torch.Tensor], [])
        patch_loc_preds = torch.jit.annotate(List[Tuple[int,int,int,int,int,int]], [])

        # Sample the image in patches
        sampler = GridSampler(img,patch_size,patch_overlap)
        patches = sampler.patches
        patches_location = sampler.patches_location
        padded_shape = sampler.padded_shape

        # Batches generation
        batches: List[torch.Tensor] = []
        batches_locations : List[List[Tuple[int,int,int,int,int,int]]] = []
        for i in range(0, len(patches), self.batch_size): # TODO : manage case where len(patches) is not a multiple of batch_size
            batch = patches[i:i + self.batch_size]
            batches.append(torch.stack(batch, dim=0))
            batches_locations.append(patches_location[i:i + self.batch_size])

        if self.num_workers <= 1 : # No need of multiprocessing
            for i in range(len(batches)):
                self.process_batch(batches[i],batches_locations[i],pred_aggr,patch_loc_preds)
        else:
            # Initialize parallelisation and run firsts workers
            next_batch_index = 0
            running_workers = torch.jit.annotate(List[Tuple[int, torch.jit.Future[None]]], []) # (index of batch, worker)
            while next_batch_index < len(batches) and len(running_workers) < self.num_workers :
                print("Batch ",next_batch_index+1,"over",len(batches))
                worker = torch.jit._fork(self.process_batch, batches[next_batch_index], batches_locations[next_batch_index],pred_aggr,patch_loc_preds)
                running_workers.append((next_batch_index, worker))
                next_batch_index += 1

            # Run remaining batches when worker available
            while len(running_workers) > 0:
                _, fut = running_workers.pop(0)  
                # Replace with ._done if torchscript ever implement it, for dynamic pool instead of queue
                torch.jit._wait(fut)        

                if next_batch_index < len(batches):
                    fut = torch.jit._fork(self.process_batch, batches[next_batch_index], batches_locations[next_batch_index],pred_aggr,patch_loc_preds)
                    running_workers.append((next_batch_index, fut))
                    next_batch_index += 1
                    

        print("Aggregation...")
        aggregator = GridAggregator(pred_aggr,patch_loc_preds,original_shape,patch_size,padded_shape)
        logit = aggregator.logit
        logit = PostProcessing.seg_postprocessing(logit,self.use_softmax,self.force_softmax,self.return_logit,self.keep_big_only,self.keep_biggest_only)

        return logit