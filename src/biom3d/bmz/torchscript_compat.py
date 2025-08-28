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
        *GridSampler*

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
        *_pad*

        Apply symetric pad so patch stay in the image bound
        
        Returns
        ----------
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
        *_lexsort_tensor*

        Torchscript friendly (and adaptated to our format) version of lexsort then toList().

        Parameters
        ----------
        data: torch.Tensor
            The data to sort

        Returns
        ----------
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
    
    def _compute_location(self):
        """
        *_compute_location*

        Compute location of patches and return them in a deterministicly sorted list
          
        Returns
        ----------
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
        *_crop*

        Generate the patch at given location by cropping `self._img`.

        Parameters
        ----------
        index_start : Tuple[int,int,int])
            Stating indexes of the patch.
          
        Returns
        ----------
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
        *_generate_patch*

        Generate the patches with `self.patches_location`.
          
        Returns
        ----------
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
        *GridAggergator*

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

    def _get_hann_window(self):
        """
        *_get_hann_window*

        Get the hann window needed for aggregation

        Returns 
        ----------
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
        *_add_patch_hann*
        
        Add the patch to the logit, ponderating with an hann window

        Parameters 
        ----------
        patch : torch.Tensor
            The patch to had
        location: Tuple[int,int,int,int,int,int]
            The location of the patch
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
        *_finalize_output*
        
        Compute the mean of the logit by the average mask tensor

        Returns 
        ----------
        The logit, still in padded size
        """
        return torch.true_divide(self.logit, self._avgmask_tensor)
    
    def _crop(self,border: torch.Tensor) -> torch.Tensor:
        """
        *_crop*
        
        Crop symetrically the logit by the `border`

        Parameters 
        ----------
        border : torch.Tensor
            A tensor of size 3 that describe by how much the padding must be done it the 3 spatial dimension.

        Returns 
        ----------
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
        """
        v = v2 - v1
        return torch.sqrt(torch.sum(v * v)).item()
    
    @staticmethod
    def _center(labels:torch.Tensor, idx:int):
        """
        return the barycenter of the pixels of label = idx
        """
        
        return torch.mean((torch.argwhere(labels.to(torch.float32) == float(idx))).to(torch.float32), dim=0)
    
    @staticmethod
    def _otsu_thresholding(im):
        """Otsu's thresholding.
        """
        threshold_range = torch.linspace(im.min(), im.max()+1, steps=255)
        criterias = torch.tensor([SizeFilter._compute_otsu_criteria(im, th) for th in threshold_range])
        best_th = threshold_range[torch.argmin(criterias,dim=None)]
        return best_th
    
    @staticmethod
    def _compute_otsu_criteria(im, th):
        """Otsu's method to compute criteria.
        Found here: https://en.wikipedia.org/wiki/Otsu%27s_method
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
    def _volumes(labels):
        """
        returns the volumes of all the labels in the image
        """
        return torch.unique(labels, return_counts=True,dim=None)[1]
    
    @staticmethod
    def _get_neighbors(z: int, y: int, x: int,D:int,H:int,W:int) -> List[Tuple[int, int, int]]:
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
    def _label(msk)->Tuple[torch.Tensor,int]:
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
    def _keep_center_only(msk):        
        """
        return mask (msk) with only the connected component that is the closest 
        to the center of the image.
        """
        labels, num = SizeFilter._label(msk)
        close_idx = SizeFilter._closest(labels,num)
        return (labels==close_idx).astype(msk.dtype)*255
    
    @staticmethod
    def _closest(labels:torch.Tensor, num:int):
        """
        return the index of the object the closest to the center of the image.
        num: number of label in the image (background does not count)
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
    def keep_big_volumes(msk, thres_rate:float=0.3):
        """
        Return the mask (msk) with less labels/volumes. Select only the biggest volumes with
        the following strategy: minimum_volume = thres_rate * np.sum(np.square(vol))/np.sum(vol)
        This computation could be seen as the expected volume if the variable volume follows the 
        probability distribution: p(vol) = vol/np.sum(vol) 
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
    def keep_biggest_volume_centered(msk):
        """
        return mask (msk) with only the connected component that is the closest 
        to the center of the image if its volumes is not too small ohterwise returns
        the biggest object (different from the background).
        (too small meaning that its volumes shouldn't smaller than half of the biggest one)
        the final mask intensities are either 0 or msk.max()
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

    def __init__(self,
                model,
                original_shape,
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

    def process_batch(self,X:torch.Tensor,batch_locations:List[Tuple[int,int,int,int,int,int]],pred_aggr:List[torch.Tensor],patch_loc_preds:List[Tuple[int,int,int,int,int,int]]):
        with torch.no_grad():
            if self.device == 'cuda':
                X = X.cuda()
            
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

    def forward(self,img:torch.Tensor):
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

        if self.num_workers <= 1 :
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