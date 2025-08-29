"""
The main difficulty for export is to save the model, saving as torch doesn't work as our model use a double patching system 
(the one in the neural network) and the one of predictor that break the need of a precise input shape, however it isn't directly compatible with
bioimage.io consumer and so we use torchscript

There are 3 problems with torschript:
- The compiler is a pain
- Adding new model is a pain
- Everything must be static


For the moment only VGG3dDeep work, and only the backup in biom3d.bmz.models, the two modification are type hinting (torchscript doesn't want 
thing like torch.nn.Module,...) and the self.head that isn't explicitly declared.

In Biapy they use a Softmax layer in model to avoid the former missing softmax post processing, so maybe a similar approach can allow us to avoid torchscript.

Need to test if recent version of models are ONNX compatible that would be better.
"""

import os
import shutil
import hashlib
from typing import Any, Optional
import warnings
import yaml
from datetime import datetime, timezone
from importlib import resources

import matplotlib.pyplot as plt
import numpy as np
import torch

import bioimageio.spec.model.v0_5 as bmz
from bioimageio.spec import save_bioimageio_package,dump_description
from bioimageio.spec.model.v0_5 import LinkedModel,ReproducibilityTolerance,Author,CiteEntry,Uploader,Maintainer,LinkedDataset

from biom3d.bmz.torchscript_compat import ModelExport
from biom3d import utils
from biom3d.builder import Builder, read_config
from biom3d.preprocess import seg_preprocessor
from biom3d.utils.config import AttrDict


class Loader(Builder):
    """
    This class is an extension of biom3d.builder.Builder that only load a model without doing any prediction or training.
    
    :ivar AttrDict config: the content of the config file as a dict
    :ivar torch.nn.Module model: The neural network
    """

    def __init__(self, path):                 
        path_to_config = os.path.join(path, 'log','config.yaml')
        self.config = utils.adaptive_load_config(path_to_config)

        # if cuda is not available then deactivate USE_FP16
        if not torch.cuda.is_available():
            self.config.USE_FP16 = False

        # load the model weights
        model_dir = os.path.join(path, 'model')
        model_name = utils.adaptive_load_config(os.path.join(path,"log","config.yaml")).DESC+'.pth'
        ckpt_path = os.path.join(model_dir, model_name)
        if not torch.cuda.is_available():
            ckpt = torch.load(ckpt_path,weights_only=True,map_location=torch.device('cpu'))
        else:
            ckpt = torch.load(ckpt_path,weights_only=True)
        print("Loading model from", ckpt_path)

        # For now only VGG3dDeep is 100% working with torchsript, and only a slighly modified version 
        if self.config.MODEL.fct == "UNet3DVGGDeep":
            from biom3d.bmz.models.unet3d_vgg_deep import UNet
            self.model = read_config(self.config.MODEL,AttrDict(UNet3DVGGDeep=AttrDict(fct=UNet, kwargs=AttrDict())))
        else : raise NotImplementedError("Only VGG3DDeep has torchscript compatibility and thus can be exported.\n"\
                                         "If the model it torchscript compatible, add it to biom3d.bmz.package.Loader class.")
        print(self.model.load_state_dict(ckpt['model'], strict=False))

def to_torchscript(img:torch.Tensor,axes_order:str,loader:Loader)->torch.jit.RecursiveScriptModule:
    """
    Convert a model to torchscript.

    Parameters
    ----------
    img: torch.Tensor
        A test image that is representative for the model.
    axes_order:str
        An axes order for the model's images. Eg: CZYX
    loader: biom3d.Loader
        A Loader instance used to load and build the model

    Returns
    -------
    torch.jit.RecursiveScriptModule
        The model as a torchscript script

    Notes
    -----
    If the package fails, it is 99% sure that this the function that crash.
    
    """
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available():device="mps"
    else: device = 'cpu'

    num_worker=4 #TODO: adapt depending on image size
    batch_size=2 #TODO: adapt depending on image size

    model = ModelExport(loader.model,
                        img.shape,
                        axes_order,
                        loader.config["PREDICTOR"]["kwargs"]["patch_size"],
                        num_workers=num_worker,
                        device=device,
                        tta=loader.config["PREDICTOR"]["kwargs"]["tta"],
                        batch_size=batch_size,
                        use_softmax=loader.config["POSTPROCESSOR"]["kwargs"]["use_softmax"],
                        force_softmax=loader.config["POSTPROCESSOR"]["kwargs"]["force_softmax"] if "force_softmax" in loader.config["POSTPROCESSOR"]["kwargs"].keys() else False,
                        keep_big_only=loader.config["POSTPROCESSOR"]["kwargs"]["keep_big_only"],
                        keep_biggest_only=loader.config["POSTPROCESSOR"]["kwargs"]["keep_biggest_only"],
                        return_logit=loader.config["POSTPROCESSOR"]["kwargs"]["return_logit"] if "return_logit" in loader.config["POSTPROCESSOR"]["kwargs"].keys() else False,
                        )
    return torch.jit.script(model)

def compute_file_hash(file_path:str, algorithm:str='sha256')->str:
    """
    Compute the hash of a file using the specified algorithm.

    Stolen at https://www.geeksforgeeks.org/python-program-to-find-hash-of-file/

    Parameters
    ----------
    file_path: str
        Path to the file to hash
    algorithm: str, default='sha256'
        Code for the hashing algorithm

    Returns
    -------
    str
        The hash created from the file
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as file:
        while chunk := file.read(8192):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()

def save_run_mode(name:str,output:str,kwargs:dict[str,Any])->None:
    """"
    Save a run-mode field as a yaml, after validating it.

    Parameters
    ----------
    name: str
        Name of the run-mode 
    output: str
        Path to the output yaml (with extension)
    kwargs_dict: dict from str to any
        Any possible parameter for the run-mode

    Returns
    -------
    None
    """
    data = {"name": name, "kwargs": kwargs}
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    bmz.RunMode.model_validate(data)
    with open(output, "w") as f:
            yaml.dump(data, f, sort_keys=False)
    print(f"File saved at '{output}'")

def save_config(tolerances:list[ReproducibilityTolerance],
                add_tol:dict[str,Any],
                add_conf:dict[str,Any],
                output:str
                )->None:
    """"
    Save a config field as a yaml, after validating it.

    Parameters
    ----------
    tolerances: list of ReproductibilityTolerance
        List of config `bioimageio` field reproductibility tolerance.
    add_tol: dict of str to any
        Any supplementary parameters for the `bioimageio`field
    add_conf: dict from str to any
        Any possible parameter for the config (other than `bioimageio`)
    output: str
        Path to the output yaml (with extension)

    Returns
    -------
    None
    """
    data = {
        "bioimageio": {
            "reproducibility_tolerance": [tol.model_dump() for tol in tolerances],
            **(add_tol or {})
        },
            **(add_conf or {})
    }
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    bmz.Config.model_validate(data)
    with open(output, "w") as f:
            yaml.dump(data, f, sort_keys=False)
    print(f"File saved at '{output}'")

def load_config(file:str)->bmz.Config:
    """
    Instanciate a Config object from a yaml file.

    Parameters
    ----------
    file: str
        Path to the yaml file

    Returns
    -------
    config: Config
        The validated Config instance described in the file.
    """
    with open(file, "r") as f:
        data = yaml.safe_load(f)
    config = bmz.Config.model_validate(data)
    return config

def load_run_mode(file):
    """
    Instanciate a RunMode object from a yaml file.

    Parameters
    ----------
    file: str
        Path to the yaml file

    Returns
    -------
    RunMode
        The validated RunMode instance described in the file.
    """
    with open(file, "r") as f:
        data = yaml.safe_load(f)
    bmz.RunMode.model_validate(data)
    return bmz.RunMode(name=data["name"],kwargs=data["kwargs"])

def _make_prediction(model,img,is_2d=False)->np.ndarray:
    """
    Make a prediction and ensure output is in CZYX or CYZ as numpy.ndarray.
    
    Parameters
    ----------
    model: torch.jit.RecursiveScriptModule
        The compiled torchscript module
    img: numpy.ndarray
        The input image
    is_2d: bool, default=False
        Whether image is in 2D

    Returns
    -------
    output: numpy.ndarray
        Prediction output (whether uint8 mask or float32 if return_logit)
    """
    print("Prediction started")
    output = model(torch.from_numpy(img))
    print("Prediction done")
    if is_2d and len(output.shape) == 2 : output = output.unsqueeze(0)
    if not is_2d and len(output.shape) == 3 : output = output.unsqueeze(0)
    return output.numpy()

def _get_default_doc_path(folder:str) -> str:
    """
    Get the default_doc.md from the package and save it in given folder.

    Parameters
    ----------
    folder: str 
        Path to a folder in which save the doc.

    Returns
    -------
    temp_path:str
        Path to the newly copied doc.
    """
    with resources.as_file(resources.files("biom3d.bmz").joinpath("default_doc.md")) as path:
        temp_path = os.path.join(folder,"default_doc.md")
        shutil.copy(path, temp_path)
    return temp_path

def _extract_2d_slice(volume: np.ndarray,is_2d:bool=False):
    """"
    Extract a 2D slice from a 3D image, at middle.

    Parameters
    ----------
    volume: np.ndarray
        Our 3d image
    is_2d: bool, default=False
        Whether image is in 2d, used to differentiate ZYX fril CYX

    Raises
    ------
    ValueError: if image number of dimension >4 or <2

    Returns
    -------
    slice_2d: np.ndarray
        The 2d middle slice (or input if 2D with ndim=2)
    """
    shape = volume.shape
    ndim = volume.ndim

    if ndim == 4:
        # (C,Z,Y,X)
        _, z, _, _ = shape
        slice_2d = volume[0, z//2, :, :]
    elif ndim == 3:
        if is_2d:  # (C,Y,X)
            slice_2d = volume[0, :, :]
        else:
            # (Z,Y,X)
            z, _, _ = shape
            slice_2d = volume[z//2, :, :]
    elif ndim == 2:
        slice_2d = volume
    else:
        raise ValueError(f"Unsuported format, ndim={ndim}, shape={shape}")
    
    return slice_2d

def _center_crop_2d(image: np.ndarray, size: int) -> np.ndarray:
    """
    Make a square crop around the center of a 2D image.

    Parameters
    ----------
    image: numpy.ndarray
        The image to crop
    size: int
        Size of the cut edge in pixel

    Returns
    -------
    numpy.ndarray
        The cropped image
    """
    h, w = image.shape
    top = (h - size) // 2
    left = (w - size) // 2
    return image[top:top + size, left:left + size]

def create_snapshot(raw:np.ndarray, pred:np.ndarray,is_2d:bool=False)->np.ndarray:
    """
    Create a cover image in ratio 2:1 with left part being raw and right part prediction.

    Parameters
    ----------
    raw: numpy.ndarray
        The raw image with 4,3 or 2 dimensions
    pred: numpy.ndarray
        The prediction associated with the raw, with 4,3 or 2 dimensions
    is_2d: bool, default=False
        Whether image is in 2d

    Returns
    -------
    numpy.ndarray
        A 2d image
    """
    raw_slice= _extract_2d_slice(raw,is_2d)
    pred_slice= _extract_2d_slice(pred,is_2d)
    h1, w1 = raw_slice.shape
    h2, w2 = pred_slice.shape

    s = min(h1, w1, h2, w2)

    raw_cropped = _center_crop_2d(raw_slice, s)
    pred_cropped = _center_crop_2d(pred_slice, s)

    snapshot = np.concatenate([raw_cropped, pred_cropped], axis=1)  # côte à côte

    return snapshot

def _get_input_tensor_desc(loader:Loader,axes_order:str,img:np.ndarray,folder:str)->bmz.InputTensorDescr:
    """
    Extract the bioimageio input tensor desriptor.

    Deduce preprocessing from loader.
    
    Parameters
    ----------
    loader: Loader
        The Loader used to instanciate the model. Is used to deduce preprocessing
    axes_order: str
        A string representing axes_order, eg:CZYX
    img: numpy.ndarray
        A representative input for the model
    folder: str
        Path to the folder used to create the bioimageio package.

    Returns
    -------
    bioimage.io.spec.model.v0_5.InputTensorDescr
        Input descriptor, with preprocessing, axe order and axe size.
    """
    clipping_bounds = loader.config["CLIPPING_BOUNDS"]
    preprocessing=[bmz.EnsureDtypeDescr(kwargs=bmz.EnsureDtypeKwargs(dtype="float32"))]
    if clipping_bounds != []: preprocessing.append(bmz.ClipDescr(kwargs=bmz.ClipKwargs(
        min=clipping_bounds[0],
        max=clipping_bounds[1],
        )))
    intensity_moment = loader.config["INTENSITY_MOMENTS"]
    znorm_kwargs={"eps":1e-15} # If the model fail it is because of this
    if intensity_moment != []:
        znorm_kwargs["mean"]=intensity_moment[0]
        znorm_kwargs["std"]=intensity_moment[1]
    preprocessing.append(bmz.ZeroMeanUnitVarianceDescr(kwargs=znorm_kwargs))
        


    input_axes = []
    for i in range(len(axes_order)):
        if axes_order[i]== 'c':
            identifiers = []
            for j in range(img.shape[i]):
                identifiers.append(bmz.Identifier("Channel_"+str(j)))
            input_axes.append(bmz.ChannelAxis(channel_names=identifiers))
        else :
            input_axes.append(bmz.SpaceInputAxis(id=bmz.AxisId(axes_order[i]), size=img.shape[i]))
    return [
        bmz.InputTensorDescr(
            id=bmz.TensorId("raw"),
            axes=input_axes,
            data=bmz.IntervalOrRatioDataDescr(type="float32"),
            test_tensor=bmz.FileDescr(source=bmz.RelativeFilePath(os.path.join(folder, "test_input.npy"))),
            preprocessing=preprocessing
        )
    ]

def _get_output_tensor_desc(loader:Loader,axes_order:str,output:np.ndarray,folder:str)->bmz.OutputTensorDescr:
    """
    Extract the bioimageio output tensor desriptor.

    Deduce postprocessing from loader.
    
    Parameters
    ----------
    loader: Loader
        The Loader used to instanciate the model. Is used to deduce postprocessing
    axes_order: str
        A string representing axes_order, eg:CZYX
    output: numpy.ndarray
        A representative output for the model
    folder: str
        Path to the folder used to create the bioimageio package.

    Returns
    -------
    bioimage.io.spec.model.v0_5.OutputTensorDescr
        Output descriptor, with postprocessing, axe order and axe size.
    """
    output_axes = []
    return_logit = loader.config["POSTPROCESSOR"]["kwargs"]["return_logit"] if "return_logit" in loader.config["POSTPROCESSOR"]["kwargs"].keys() else False
    output_type="uint8" if not return_logit else "float32"
    for i in range(len(axes_order)):
        if axes_order[i]== 'c':
            identifiers = []
            for j in range(output.shape[i]):
                identifiers.append(bmz.Identifier("Channel_"+str(j)))
            output_axes.append(bmz.ChannelAxis(channel_names=identifiers))
        else :
            output_axes.append(bmz.SpaceOutputAxis(id=bmz.AxisId(axes_order[i]), size=output.shape[i]))
    return [
        bmz.OutputTensorDescr(
            id=bmz.TensorId("predictions"),
            axes=output_axes,
            data=bmz.IntervalOrRatioDataDescr(type=output_type),
            test_tensor=bmz.FileDescr(source=bmz.RelativeFilePath(os.path.join(folder, "test_output.npy"))),
            postprocessing=[bmz.EnsureDtypeDescr(kwargs=bmz.EnsureDtypeKwargs(dtype=output_type))]
        )
    ]

def _load_and_preprocess(axes:Optional[str],
                         test_image:str,
                         loader:Loader
                         )->tuple[np.ndarray,str,np.ndarray]:
    """
    Load the input image, preprocess it and deduce axes order (if possible).

    Parameters
    ----------
    axes: str or None
        A string representing axes order. Will overwrite the axes order in the image metadata if not None.
    test_image: str
        Path to an image
    loader: Loader 
        Loader instance used to instanciate the model

    Raises
    ------
    AssertionError: If 'axes' key not in image metadata and axes paramter is None

    Returns:
    img: np.ndarray
        Loaded image
    axes_order: str
        The final axes order
    img_process
        The preprocessed image
    """
    img, metadata = utils.adaptive_imread(test_image)
    if axes != None:
        metadata["axes"] = axes
    assert "axes" in metadata, "Axes order can't be found, you can specify it by using the --axes argument" 
    if len(img.shape) == 3 : img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    # Axes order modification by preprocessing TODO move it in preprocessor
    axes_order = str.lower(metadata["axes"])
    axes_order = axes_order.replace('t', '') # Remove time dimension if exist
    if len(img.shape)==3 or len(axes_order)==3:
        axes_order = axes_order.replace('c', '') #Channel dimension has been ignored by numpy and will be placed in first place by preprocess
        axes_order = 'c'+ axes_order
    elif len(img.shape)==4: # Channel axes will be swaped in first place by preprocessing
        tmp = list(axes_order)
        tmp[loader.config["CHANNEL_AXIS"]] = tmp[0]
        tmp[0] = 'c'
        axes_order = ''.join(tmp) 

    # Preprocessing
    preprocessor = loader.config["PREPROCESSOR"]["kwargs"]
    # TODO replace with a more flexible approach, this work only because their is only one preprocessing function
    img_process,_ = seg_preprocessor(
                                            img,metadata,
                                            median_spacing=preprocessor['median_spacing'],
                                            clipping_bounds=preprocessor['clipping_bounds'],
                                            intensity_moments=preprocessor['intensity_moments'],
                                            channel_axis=preprocessor['channel_axis'],
                                            num_channels=preprocessor['num_channels'],
                                            num_classes=loader.config.NUM_CLASSES
                                            )   
    return img, axes_order,img_process

def _validate_inputs(descr:str, author_list:list[Author], cite_list:list[CiteEntry])->None:
    """
    Validate the mandatory inputs for a bioimageio package.

    Parameters
    ----------
    descr: str
        The model description.
    author_list: list of bioimageio.spec.model.v0_5.Author
        The author list.
    cite_list: list of bioimageio.spec.model.v0_5.CiteEntry
        The citation list.

    Raises
    ------
    ValueError: If any parameter None or empty.

    Returns
    -------
    None
    """
    if not descr:
        raise ValueError("Model description must be given.")
    if not author_list:
        raise ValueError("At least one author should be given.")
    if not cite_list:
        raise ValueError("At least one citation must be given.")

def _create_output_folder(output_folder:str, model_name:str)->str:
    """
    Create a new folder at given location to store model infos.

    Parameters
    ----------
    output_folder: str
        The path of the folder in which the new foler will be created
    model_name: str
        The model name, used to name the new folder

    Returns
    -------
    folder: str
        Path to newly created folder : <output_folder>/<model_name>_tmp
    """
    folder = os.path.join(output_folder, f"{model_name}_tmp")
    os.makedirs(folder, exist_ok=True)
    print("Folder created at", folder)
    return folder

def _archive_original_model(path_to_model:str, folder:str)->str:
    """
    Create a zip of the model and copy it in package folder.

    The created archive will have `Original_Biom3d_Model.zip` as name and will contains the image (if exist), log and model subfolders.

    Parameters
    ----------
    path_to_model: str
        Path to the model folder.
    folder: str
        Path to the package folder

    Returns
    -------
    zipped_name: str
        Path to the zip, <folder>/Original_Biom3d_Model.zip
    """
    zipped_name = "Original_Biom3d_Model.zip"
    zip_path = os.path.join(folder, zipped_name)
    shutil.make_archive(zip_path.replace(".zip", ""), 'zip', root_dir=path_to_model)
    return zipped_name

def _prepare_model_and_images(folder:str, 
                             path_to_model:str, 
                             test_image:str, 
                             axes:Optional[str]=None, 
                             pred:Optional[str]=None
                             )->tuple[Loader,str,str,np.ndarray,np.ndarray]:
    """
    Load the model, the input image and make/load prediction. Convert the model to torchscript.

    Parameters
    ----------
    foder: str
        Path to the package folder
    path_to_model: str
        It's quite clear
    test_image: str
        Path to a representative input image for the model.
    axes: str, optional
        String represneting the input image axes order (ef:CZYX), if None, 'axes' should be in image meta_data.
    pred: str, optional
        Path to the prediction associated with test_image. If None, the prediction will be made.

    Returns
    -------
    loader: Loader
        A Loader instance encapsulating the model and its config.
    modl_sha: str
        The sha256 digest of the torchsript weights
    axes_order: str
        Axes order from inputs
    img_process: numpy.ndarray
        Prepropressed image, used later to describe input descriptor (for shape and axes)
    output: numpy.ndarray
        Model output image, used later to describe output descriptor (for shape and axes)
    """
    loader = Loader(path_to_model)
    img, axes_order, img_process = _load_and_preprocess(axes, test_image, loader)
    np.save(os.path.join(folder, 'test_input.npy'), img, allow_pickle=False)

    model = to_torchscript(img, axes_order, loader)
    model_path = os.path.join(folder, "weights-torchscript.pt")
    model.save(model_path)
    model_sha = compute_file_hash(model_path)

    if pred is None:
        output = _make_prediction(model, img_process,loader.config.IS_2D)
    else:
        output = utils.adaptive_imread(pred)[0]

    np.save(os.path.join(folder, 'test_output.npy'), output, allow_pickle=False)

    return loader, model_sha, axes_order, img_process, output

def _handle_cover_image(folder:str, cover:Optional[str], img:Optional[np.ndarray], output:Optional[np.ndarray], is_2d:bool)->str:
    """
    Create or load a cover image and transfer it to package folder.

    Parameters
    ----------
    folder: str
        Path to package folder.
    cover: str or None
        Path to a cover image. If None img, output and is_2d will be used to create one.
    img: np.ndarray, optional
        Input image used to generate cover if cover is None
    output: np.ndarray, optional
        Output image used to generate cover if cover is None
    is_2d: bool
        Wheter images are 2D, used to generate cover if cover is None

    Raises
    ------
    FileNotFoundError: If cover is not None and file doesn't exist.

    Returns
    -------
    cover_path: str
        Path to the cover: <folder>/cover.png
    """
    cover_path = os.path.join(folder, "cover.png")
    if cover is not None:
        if not os.path.exists(cover):
            raise FileNotFoundError("Cover image not found.")
        shutil.copy(cover, cover_path)
    else:
        snapshot = create_snapshot(img, output, is_2d)
        plt.imsave(cover_path, snapshot)
    return cover_path

def _build_rdf_kwargs(folder:str, 
                     model_name:str, 
                     descr:str, 
                     author_list:list[Author], 
                     cite_list:list[CiteEntry], 
                     model_sha:str,
                     axes_order:str, 
                     img_process:np.ndarray, 
                     loader:Loader, 
                     output:np.ndarray, 
                     doc_file:str, 
                     cover_path:str,
                     license:Optional[str], 
                     tags:list[str], 
                     git:Optional[str], 
                     training_data:Optional[LinkedDataset], 
                     parent:Optional[LinkedModel], 
                     version:Optional[str],
                     version_comment:Optional[str], 
                     uploader:Optional[Uploader], 
                     run_mode:Optional[str], 
                     config:Optional[str], 
                     packagers_list:Optional[list[Author]],
                     maintainers_list:Optional[list[Maintainer]], 
                     links:Optional[list[str]], 
                     zipped:str, 
                     attachments:Optional[list[str]]
                     )->dict[str,Any]:
    """
    Create a parameter dictionary depending on the input and ready to use for instantiating a ModelDescr.
    
    Parameters
    ----------
    folder: str
        Path to package folder
    model_name: str
        Name of the model, used to name the package
    descr: str
        Short description for the model
    author_list: list of bioimageio.spec.model.v0_5.Author
        List of model authors, not empty
    cite_list: list of bioimageio.spec.model.v0_5.CiteEntry
        List of model citation, not empty
    model_sha: str
        Sha256 hash of torchscript weights
    axes_order: str
        String representing inputs axes order (eg: CZYX).
    imp_process: numpy.ndarray
        Preprocessed image, used to determine input shape, axes and datatype
    loader: Loader
        Loader instance that encapsulate the model
    output: numpy.ndarray
        Output image, used to determine output shape, axes and datatype
    doc_file: str
        Path to the model documentation in markdown
    cover_image: str
        Path to the model cover image
    license: str or None
        Code corresponding to the model license. If None, MIT license is applied.
    tags: list of str
        List of tag to help the model referencing
    git: str or None
        Url to the model git repo
    training_data: bioimageio.spec.model.v0_5.LinkedDataset or None
        Bioimageio dataset used for training, if exists. 
    parent: bioimageio.spec.model.v0_5.LinkedModel or None
        Bioimageio model used as base, if fine tuning.
    version: str or None
        Model version
    version_comment: str or None
        A short version description
    uploader: bioimageio.spec.model.v0_5.Uploader or None
        The person who uploaded the model. Relevant only if not one of the authors.
    run_mode: str or None
        Path to a yaml file describing a bioiageio.spec.model.v0_5.RunMode
    config: str or None
        Path to a yaml file describing a bioiageio.spec.model.v0_5.Config
    packager_list: list of bioiageio.spec.model.v0_5.Author or None
        List of persons that packages the model. Relevant only if not in the authors.
    maintainer_list: list of bioimageio.spec.model.v0_5.Maintainer or None
        List of persons that maintain the git repository. Relevant only if git is not None.  
    links: list of str or None
        List of strings representing other biomimageio ressources id (eg: 'deepimagej/deepimagej')
    zipped: str
        Path to the zipped model.
    attachment: list of str
        List of relative path to any other file that should be attached.

    Returns
    -------
    rdf_kwargs: dict from str to any
        A ready to use parameter dictionary for bioimageio.spec.model.v0_5.ModelDescr without any superflu parameter.
    """
    rdf_kwargs = {
        "name": model_name,
        "description": descr,
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "authors": author_list,
        "cite": cite_list,
        "documentation": bmz.RelativeFilePath(doc_file),
        "covers": [bmz.RelativeFilePath(cover_path)],
        "inputs": _get_input_tensor_desc(loader, axes_order, img_process, folder),
        "outputs": _get_output_tensor_desc(loader, axes_order, output, folder),
        "weights": bmz.WeightsDescr(
            torchscript=bmz.TorchscriptWeightsDescr(
                source=bmz.RelativeFilePath(os.path.join(folder, "weights-torchscript.pt")),
                sha256=bmz.Sha256(model_sha),
                pytorch_version=bmz.Version(torch.__version__)
            )
        ),
        "license": bmz.LicenseId("MIT") if license is None else bmz.LicenseId(license),
    }

    # Optionals
    if tags: rdf_kwargs["tags"] = tags
    if git: rdf_kwargs["git_repo"] = git
    if training_data: rdf_kwargs["training_data"] = training_data
    if parent: rdf_kwargs["parent"] = parent
    if version: rdf_kwargs["version"] = version
    if version_comment: rdf_kwargs["version_comment"] = version_comment
    if uploader: rdf_kwargs["uploader"] = uploader
    if run_mode: rdf_kwargs["run_mode"] = load_run_mode(run_mode)
    if config: rdf_kwargs["config"] = load_config(config)
    if packagers_list: rdf_kwargs["packaged_by"] = packagers_list
    if maintainers_list: rdf_kwargs["maintainers"] = maintainers_list
    if links: rdf_kwargs["links"] = links

    # Attachments
    attach = [bmz.FileDescr(source=bmz.RelativeFilePath(os.path.join(folder, zipped)))]
    if attachments:
        for e in attachments:
            path = os.path.join(folder, os.path.basename(e))
            shutil.copy(e, path)
            attach.append(bmz.FileDescr(source=bmz.RelativeFilePath(path)))
    rdf_kwargs["attachments"] = attach

    return rdf_kwargs

def package_bioimage_io(path_to_model:str,
                        test_image:str,
                        doc_file:str,
                        descr:str,
                        author_list:list[Author],
                        cite_list:list[CiteEntry],
                        output_folder:Optional[str]=None,
                        axes:Optional[str]=None,
                        model_name:str="Unet_Biom3d",
                        cover:Optional[str]=None,
                        license:Optional[str]=None,
                        tags:Optional[list[str]]=None,
                        git:Optional[str]=None,
                        attachments:Optional[list[str]]=None,
                        version:Optional[str]=None,
                        version_comment:Optional[str]=None,
                        uploader:Optional[str]=None,
                        maintainers_list:Optional[list[Maintainer]]=None,
                        packagers_list:Optional[list[Author]]=None,
                        training_data:Optional[LinkedDataset]=None,
                        parent:Optional[LinkedModel]=None,
                        links:Optional[list[str]]=None,
                        run_mode:Optional[str]=None,
                        config:Optional[str]=None,
                        keep_dir:bool=False,
                        pred:Optional[str]=None,
                        ):
    """
    Create a parameter dictionary depending on the input and ready to use for instantiating a ModelDescr.
    
    Parameters
    ----------
    path_to_model: str
        Path to model folder
    test_image: str
        Path to a representative input for the model
    doc_file: str
        Path to the model documentation in markdown
    descr: str
        Short description for the model
    author_list: list of bioimageio.spec.model.v0_5.Author
        List of model authors, not empty
    cite_list: list of bioimageio.spec.model.v0_5.CiteEntry
        List of model citation, not empty
    output_folder: str, optional
        Path to a folder where the temporary package folder will be created
    axes: str, optional
        String representing inputs axes order (eg: CZYX). If None, 'axes' need to be in image metadata.
    model_name: str, default= "Unet_Biom3d"
        Name of the model, used to name the package
    cover: str, optional
        Path to a cover image, must be in 2:1. If None, a cover image will be automatically generated.
    license: str, optional
        License code for the model, see complete list here : https://bioimage-io.github.io/spec-bioimage-io/bioimageio_schema_latest/#oneOf_i2_oneOf_i1_license. If None, MIT is applied.
    tags: list of str, optional
        List of tag to help the model referencing, altough optional, it is highly recommended to put some.
    git: str, optional
        Url to the model git repo
    attachment: list of str, optional
        List of relative path to any other file that should be attached. By default, only the zipped original biom3d model is attached.
    version: str, optional
        Model version. Recommended for FAIR.
    version_comment: str, optional
        A short version description
    uploader: bioimageio.spec.model.v0_5.Uploader, optional
        The person who uploaded the model. Relevant only if not one of the authors.
    maintainer_list: list of bioimageio.spec.model.v0_5.Maintainer, optional
        List of persons that maintain the git repository. Relevant only if git is not None.  
    packager_list: list of bioiageio.spec.model.v0_5.Author, optional
        List of persons that packages the model. Relevant only if not in the authors.
    training_data: bioimageio.spec.model.v0_5.LinkedDataset, optional
        Bioimageio dataset used for training, if exists. 
    parent: bioimageio.spec.model.v0_5.LinkedModel, optional
        Bioimageio model used as base, if fine tuning.
    links: list of str, optional
        List of strings representing other biomimageio ressources id (eg: 'deepimagej/deepimagej')
    run_mode: str, optional
        Path to a yaml file describing a bioiageio.spec.model.v0_5.RunMode. Can be generated with `python -m biom3d.bmz.cli run-mode`.
    config: str, optional
        Path to a yaml file describing a bioiageio.spec.model.v0_5.Config. Can be generated with `python -m biom3d.bmz.cli run-mode`.
    keep_dir: bool, default=False
       Whether to keep package folder
    pred: str, optional
        Path to the prediction associated with test_img, if None, a prediction is performed

    Returns
    -------
    None

    Notes
    -----
    The package folder doesn't contain the rdf
    """
    _validate_inputs(descr, author_list, cite_list)

    folder = _create_output_folder(output_folder, model_name)
    zipped = _archive_original_model(path_to_model, folder)

    loader, model_sha, axes_order, img_process, output = _prepare_model_and_images(
        folder, path_to_model, test_image, axes, pred
    )

    if doc_file is None:
        doc_file = _get_default_doc_path(folder)

    cover_path = _handle_cover_image(folder, cover, img_process, output, loader.config.IS_2D)

    rdf_kwargs = _build_rdf_kwargs(
        folder, model_name, descr, author_list, cite_list, model_sha,
        axes_order, img_process, loader, output, doc_file, cover_path,
        license, tags, git, training_data, parent, version,
        version_comment, uploader, run_mode, config, packagers_list,
        maintainers_list, links, zipped, attachments
    )

    my_model = bmz.ModelDescr(**rdf_kwargs)
    print("Model generated")

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with open(os.path.join(folder,"rdf.yaml"), "w") as f:
            data= dump_description(my_model)
            yaml.dump(data, f, sort_keys=False)
        save_bioimageio_package(my_model, output_path=os.path.join(output_folder, model_name.replace(" ", "_") + ".zip"))
        for w in caught_warnings:
            if "TimeOutputAxis" not in str(w.message):
                print(w.message)

    if not keep_dir:
        shutil.rmtree(folder)

    print('Model exported as', model_name.replace(" ", "_") + '.zip')

        
