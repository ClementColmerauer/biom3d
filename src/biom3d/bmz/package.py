import matplotlib.pyplot as plt
import shutil
import os
from datetime import datetime,timezone
import torch
import numpy as np
import hashlib
import yaml

from importlib import resources
 
import bioimageio.spec.model.v0_5 as bmz

from biom3d.bmz.torchscript_compat import ModelExport
from biom3d import utils
from biom3d.builder import Builder, read_config
from biom3d.preprocess import seg_preprocessor
from biom3d.utils.config import AttrDict

# TODO shatter this file

class Loader(Builder):
    def __init__(self, path):                
        # for training or fine-tuning:
        # load the config file and change some parameters if multi-gpus training    
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
        if self.config.MODEL.fct == "UNet3DVGGDeep":
            from biom3d.bmz.models.unet3d_vgg_deep import UNet
            self.model = read_config(self.config.MODEL,AttrDict(UNet3DVGGDeep=AttrDict(fct=UNet, kwargs=AttrDict())))
        else : raise NotImplementedError("Only VGG3DDeep has torchscript compatibility and thus can be exported.\n"\
                                         "If the model it torchscript compatible, add it to biom3d.bmz.package.Loader class.")
        print(self.model.load_state_dict(ckpt['model'], strict=False))

def to_torchscript(img,axes_order,loader):
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

# Stolen at https://www.geeksforgeeks.org/python-program-to-find-hash-of-file/
def compute_file_hash(file_path, algorithm='sha256'):
    """Compute the hash of a file using the specified algorithm."""
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as file:
        # Read the file in chunks of 8192 bytes
        while chunk := file.read(8192):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()

def save_run_mode(name,output,kwargs):
    data = {"name": name, "kwargs": kwargs}
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    bmz.RunMode.model_validate(data)
    with open(output, "w") as f:
            yaml.dump(data, f, sort_keys=False)
    print(f"File saved at '{output}'")

def save_config(tolerances,add_tol,add_conf,output):
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

def load_config(file):
    with open(file, "r") as f:
        data = yaml.safe_load(f)
    config = bmz.Config.model_validate(data)
    return config

def load_run_mode(file):
    with open(file, "r") as f:
        data = yaml.safe_load(f)
    bmz.RunMode.model_validate(data)
    return bmz.RunMode(name=data["name"],kwargs=data["kwargs"])

def make_prediction(model,img):
    print("Prediction started")
    output = model(torch.from_numpy(img))
    print("Prediction done")
    if len(output.shape) == 3 : output = output.unsqueeze(0)
    print(output.shape, output.numpy().shape)
    return output.numpy().astype(np.uint8)

def get_default_doc_path(folder) -> str:
    with resources.as_file(resources.files("biom3d.bmz").joinpath("default_doc.md")) as path:
        temp_path = os.path.join(folder,"default_doc.md")
        shutil.copy(path, temp_path)
        return temp_path

def extract_2d_slice(volume: np.ndarray,is_2d:bool=False):
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
        raise ValueError(f"Format non supporté, ndim={ndim}, shape={shape}")
    
    return slice_2d

def center_crop_2d(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape
    top = (h - size) // 2
    left = (w - size) // 2
    return image[top:top + size, left:left + size]

def create_snapshot(raw, pred,is_2d=False):
    raw_slice= extract_2d_slice(raw,is_2d)
    pred_slice= extract_2d_slice(pred,is_2d)
    h1, w1 = raw_slice.shape
    h2, w2 = pred_slice.shape

    s = min(h1, w1, h2, w2)

    raw_cropped = center_crop_2d(raw_slice, s)
    pred_cropped = center_crop_2d(pred_slice, s)

    snapshot = np.concatenate([raw_cropped, pred_cropped], axis=1)  # côte à côte

    return snapshot

def _get_input_tensor_desc(loader,axes_order,img,folder):
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

def _get_output_tensor_desc(loader,axes_order,output,folder):
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

def _load_and_preprocess(axes,test_image,loader):
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
def validate_inputs(descr, author_list, cite_list):
    if not descr:
        raise ValueError("Model description must be given.")
    if not author_list:
        raise ValueError("At least one author should be given.")
    if not cite_list:
        raise ValueError("At least one citation must be given.")


def create_output_folder(output_folder, model_name):
    folder = os.path.join(output_folder, f"{model_name}_tmp")
    os.makedirs(folder, exist_ok=True)
    print("Folder created at", folder)
    return folder


def archive_original_model(path_to_model, folder):
    zipped_name = "Original_Biom3d_Model.zip"
    zip_path = os.path.join(folder, zipped_name)
    shutil.make_archive(zip_path.replace(".zip", ""), 'zip', root_dir=path_to_model)
    return zipped_name


def prepare_model_and_images(folder, path_to_model, test_image, axes, pred):
    loader = Loader(path_to_model)
    img, axes_order, img_process = _load_and_preprocess(axes, test_image, loader)
    np.save(os.path.join(folder, 'test_input.npy'), img, allow_pickle=False)

    model = to_torchscript(img, axes_order, loader)
    model_path = os.path.join(folder, "weights-torchscript.pt")
    model.save(model_path)
    model_sha = compute_file_hash(model_path)

    if pred is None:
        output = make_prediction(model, img_process)
    else:
        output = utils.adaptive_imread(pred)[0]

    np.save(os.path.join(folder, 'test_output.npy'), output, allow_pickle=False)

    return loader, model_sha, axes_order, img_process, output


def handle_cover_image(folder, cover, img, output, is_2d):
    cover_path = os.path.join(folder, "cover.png")
    if cover is not None:
        if not os.path.exists(cover):
            raise FileNotFoundError("Cover image not found.")
        shutil.copy(cover, cover_path)
    else:
        snapshot = create_snapshot(img, output, is_2d)
        plt.imsave(cover_path, snapshot)
    return cover_path


def build_rdf_kwargs(folder, model_name, descr, author_list, cite_list, model_sha,
                     axes_order, img_process, loader, output, doc_file, cover_path,
                     license, tags, git, training_data, parent, version,
                     version_comment, uploader, run_mode, config, packagers_list,
                     maintainers_list, links, zipped, attachments):

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


def package_bioimage_io(path_to_model,
                        test_image,
                        doc_file,
                        descr,
                        author_list,
                        cite_list,
                        output_folder=None,
                        axes=None,
                        model_name="Unet_Biom3d",
                        cover=None,
                        license=None,
                        tags=None,
                        git=None,
                        attachments=None,
                        version=None,
                        version_comment=None,
                        uploader=None,
                        maintainers_list=None,
                        packagers_list=None,
                        training_data=None,
                        parent=None,
                        links=None,
                        run_mode=None,
                        config=None,
                        keep_dir=False,
                        pred=None,
                        ):

    validate_inputs(descr, author_list, cite_list)

    folder = create_output_folder(output_folder, model_name)
    zipped = archive_original_model(path_to_model, folder)

    loader, model_sha, axes_order, img_process, output = prepare_model_and_images(
        folder, path_to_model, test_image, axes, pred
    )

    if doc_file is None:
        doc_file = get_default_doc_path(folder)

    cover_path = handle_cover_image(folder, cover, img_process, output, loader.config.IS_2D)

    rdf_kwargs = build_rdf_kwargs(
        folder, model_name, descr, author_list, cite_list, model_sha,
        axes_order, img_process, loader, output, doc_file, cover_path,
        license, tags, git, training_data, parent, version,
        version_comment, uploader, run_mode, config, packagers_list,
        maintainers_list, links, zipped, attachments
    )

    my_model = bmz.ModelDescr(**rdf_kwargs)
    print("Model generated")

    from bioimageio.spec import save_bioimageio_package
    import warnings
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        save_bioimageio_package(my_model, output_path=os.path.join(output_folder, model_name.replace(" ", "_") + ".zip"))
        for w in caught_warnings:
            if "TimeOutputAxis" not in str(w.message):
                print(w.message)

    if not keep_dir:
        shutil.rmtree(folder)

    print('Model exported as', model_name.replace(" ", "_") + '.zip')

        
