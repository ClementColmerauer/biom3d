import argparse
from bioimageio.spec.generic.v0_3 import Author,CiteEntry,Uploader,Maintainer
from bioimageio.spec.dataset.v0_3 import LinkedDataset
from bioimageio.spec.model.v0_5 import LinkedModel,ReproducibilityTolerance
from biom3d.bmz import package_bioimage_io,save_run_mode,save_config
from enum import Enum

def _parse_field(s: str) -> str | None:
        s = s.strip()
        return None if s.lower() == "none" or s == "" else s

def _parse_author_string(author_str: str):
    parts = [_parse_field(p) for p in author_str.split(",")]

    if len(parts) != 5:
        raise ValueError(
            f"Badly formated author  '{author_str}'. "
            "Expected: 5 field separated by ',' : name,github_user,affiliation,email,orcid"
        )

    name, github_user, affiliation, email, orcid = parts

    if not name:
        raise ValueError("The 'name' field is mandatory for authors.")

    return Author(
        name=name,
        github_user=github_user,
        affiliation=affiliation,
        email=email,
        orcid=orcid,
    )

def _parse_maintainer_string(maintainer_str: str):
    parts = [_parse_field(p) for p in maintainer_str.split(",")]

    if len(parts) != 5:
        raise ValueError(
            f"Badly formated maintainer  '{maintainer_str}'. "
            "Expected: 5 field separated by ',' : name,github_user,affiliation,email,orcid"
        )

    name, github_user, affiliation, email, orcid = parts

    if not github_user:
        raise ValueError("The 'github_user' field is mandatory for maintainers.")

    return Maintainer(
        name=name,
        github_user=github_user,
        affiliation=affiliation,
        email=email,
        orcid=orcid,
    )

def _parse_cite_string(cite_str: str):
    parts = [_parse_field(p) for p in cite_str.split(",")]

    if len(parts) != 3:
        raise ValueError(
            f"Badly formated citation  '{cite_str}'. "
            "Expected: 3 field separated by ',' : text,doi,url"
        )

    text,doi,url = parts

    if not text:
        raise ValueError("The 'text' field is mandatory for authors.")
    if not doi and not url:
        raise ValueError("Either doi or url must be given.")

    return CiteEntry(
        text=text,
        url=url,
        doi=doi,
    )

def _parse_uploader_string(uploader_str: str):
    parts = [_parse_field(p) for p in uploader_str.split(",")]

    if len(parts) != 2:
        raise ValueError(
            f"Badly formated citation  '{uploader_str}'. "
            "Expected: 2 field separated by ',' : email,name"
        )

    email,name = parts

    if not email:
        raise ValueError("The 'email' field is mandatory for uploader.")

    return Uploader(
        email=email,
        name=name,
    )

def _parse_training_data_string(data_str: str):
    parts = [_parse_field(p) for p in data_str.split(",")]

    if len(parts) != 2:
        raise ValueError(
            f"Badly formated training data  '{data_str}'. "
            "Expected: 2 field separated by ',' : id,version"
        )

    training_data_id,version, = parts

    if not training_data_id:
        raise ValueError("The 'id' field is mandatory for training data.")
    if not version:
        raise ValueError("The 'version' field is mandatory for training data.")

    return LinkedDataset(
        id=training_data_id,
        version=version,
    )

def _parse_parent_string(parent_str: str):
    parts = [_parse_field(p) for p in parent_str.split(",")]

    if len(parts) != 2:
        raise ValueError(
            f"Badly formated training data  '{parent_str}'. "
            "Expected: 2 field separated by ',' : id,version"
        )

    parent_id,version, = parts

    if not parent_id:
        raise ValueError("The 'id' field is mandatory for parent.")
    if not version:
        raise ValueError("The 'version' field is mandatory for parent.")

    return LinkedModel(
        id=parent_id,
        version=version,
    )

def _parse_tolerance_args(tolerance_args):
    """
    Parse a list of --tolerance key=value arguments into a dict validated by ReproducibilityTolerance.
    """
    tolerances = []
    for arg in tolerance_args:
        # On peut parser des groupes clé=valeur multiples séparés par ';'
        entries = arg.split(";")
        tol_dict = {}
        for entry in entries:
            if '=' not in entry:
                raise ValueError(f"Format invalide (pas de '=') : {entry}")
            key, value = entry.split('=', 1)
            tol_dict[key] = _parse_value(value)
        # Valide et crée une instance ReproducibilityTolerance
        if("weights_formats") not in tol_dict: tol_dict["weights_formats"]=[] # avoid tuple
        if("output_ids") not in tol_dict: tol_dict["output_ids"]=[]
        tol = ReproducibilityTolerance(**tol_dict)
        tolerances.append(tol)
    return tolerances

def _parse_value(value: str):
    if value.startswith("list:"):
        list_str = value[len("list:"):]
        return [_parse_value(v) for v in list_str.split(",")]
    elif value.startswith("dict:"):
        dict_str = value[len("dict:"):]
        d = {}
        for item in dict_str.split(";"):
            if "=" not in item:
                raise ValueError(f"Invalid dict item (missing '='): {item}")
            k, v = item.split("=", 1)
            d[k] = _parse_value(v)
        return d
    elif value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value
        
def _parse_kwargs_list(kwargs_list):
    parsed = {}
    for entry in kwargs_list:
        if '=' not in entry:
            raise ValueError(f"Invalid format (missing '='): {entry}")
        key, value = entry.split('=', 1)
        parsed[key] = _parse_value(value)
    return parsed

class Target(Enum ):    
    v0x5BIIO = "v0.5BioImageIo"

    def __str__(self):
        return self.value

if __name__=='__main__':
    # Main parser
    parser = argparse.ArgumentParser(description="Bioimage.io command line interface for full packaging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Parser to generate a run-mode yaml
    runmode_parser = subparsers.add_parser("run-mode", help="Create a run mode parameter as a yaml file.")
    runmode_parser.add_argument("-n","--name", type=str,default='deepimagej',
                        help="Runmode name, based on https://bioimage-io.github.io/spec-bioimage-io/bioimageio_schema_latest/#tab-pane_oneOf_i2_oneOf_i1_run_mode_anyOf_i0_name_anyOf_i0") 
    runmode_parser.add_argument("-o","--output", type=str,default='./run_mode.yaml',
                        help="Path to output file, will overwrite, default is run_mode.yaml in current folder.") 
    runmode_parser.add_argument("--kwargs", action="append",type=str, required=True,
                        help="An argument in format 'key=value', with key being a string and value a string, number, boolean, list (eg list:val1,val2) or dictionary 'dict:key1=value1;key2=value2'. Can be repeated") 

    # Parser to generate a config yaml, that define reproductibility tolerance
    config_parser = subparsers.add_parser("config", help="Create config parameter as a yaml with reproducibility_tolerance")
    config_parser.add_argument("-o","--output", type=str,default='./config.yaml',
                        help="Path to output file, will overwrite, default is config.yaml in current folder.") 
    config_parser.add_argument("--additional_config", action="append",type=str,
                    help="An argument in format 'key=value', with key being a string and value a string, number, boolean, list (eg list:val1,val2) or dictionary 'dict:key1=value1;key2=value2'. To be used by something else than bioimageio. Can be repeated") 
    config_parser.add_argument("--additional_tolerance", action="append",type=str,
                    help="An argument in format 'key=value', with key being a string and value a string, number, boolean, list (eg list:val1,val2) or dictionary 'dict:key1=value1;key2=value2' It will be added to bioimage field. Can be repeated") 
    config_parser.add_argument("--tolerance", action="append", required=True,
                    help="An argument in format 'key1=value1;key2=value2', with key being a string and value a string, number, boolean, list (eg list:val1,val2) or dictionary 'dict:key1=value1;key2=value2' It will be added to bioimage field, refer to https://bioimage-io.github.io/spec-bioimage-io/bioimageio_schema_latest/#oneOf_i2_oneOf_i1_config_bioimageio_reproducibility_tolerance_items_output_ids. for possible key/values. Keys that are not included be set to theyr default values. Can be repeated") 

    # Packageing parser
    package_parser = subparsers.add_parser("package", help="Package the model.")
    package_parser.add_argument("-m","--model", type=str,required=True,
                        help="Relative path to model directory")  
    package_parser.add_argument("-i","--img",type=str,required=True,
                        help="Relative path to test image (must be tif, nifty or numpy)")  
    package_parser.add_argument("--doc",type=str,default=None,dest="doc",
                        help="Path to documentation file (markdown), default=biom3d.bmz.default_doc.md")
    package_parser.add_argument("--desc",type=str,default=None,dest="descr",required=True,
                        help="Quick description of the model (e.g : Segmentation model for mouse embryo segmentation)")
    package_parser.add_argument("-l","--license",type=str,default='MIT',dest="license",
                        help="Identifier of the license, see complete list https://bioimage-io.github.io/spec-bioimage-io/bioimageio_schema_latest/index.html#oneOf_i0_oneOf_i0_license, default is 'MIT'")
    package_parser.add_argument("--author", action="append",required=True,
                        help="Author as format: 'name,github_user,affiliation,email,orci', name is mandatory, other field can be written as 'None' and will be ignored. Can be repeated")
    package_parser.add_argument("--cite", action="append",required=True,
                        help="Citation as format: 'text,doi,url', text and either url or doi are mandatory, other field can be written as 'None' or '' and will be ignored. Can be repeated")
    package_parser.add_argument("--t", action="append",dest="tags",
                        help="Tag to include for the model, used for model referencing. Can be repeated")
    package_parser.add_argument("-a", "--axes", type = str, default = None, 
                        help="Specified axes order for images, in case it cannot be deduced from file.")
    package_parser.add_argument("--pred", type=str, default=None,
                        help="(Optional) Path to the prediction associated with img, if not provided, a prediction will be done.")
    package_parser.add_argument("-t", "--target", type=Target, default=Target.v0x5BIIO, choices=list(Target),
                        help="(Optional) Target image and version")
    package_parser.add_argument("-o", "--output_dir", type=str, default="./",
                        help="(Optional) Relative path to directory where you want your model, will create a sub folder (default local directory)")
    package_parser.add_argument("--run_mode", type=str, default=None,
                        help="(Optional) Relative path to run mode yaml file, can be created manualy or with biom3d.bmz.cli run-mode command.")
    package_parser.add_argument("--config", type=str, default=None,
                        help="(Optional) Relative path to config yaml file, can be created manualy or with biom3d.bmz.cli config command.")
    package_parser.add_argument("--training_data",type=str,default=None,dest="training_data",
                        help="(Optional) Bioimage.io dataset as format 'id,version'.")
    package_parser.add_argument("--parent",type=str,default=None,dest="parent",
                        help="(Optional) Bioimage.io model as format 'id,version'.")
    package_parser.add_argument("-v","--version",type=str,default=None,dest="version",
                        help="(Optional) Model version.")
    package_parser.add_argument("--version_comment",type=str,default=None,dest="version_comment",
                        help="(Optional) Comment describing actual version.")
    package_parser.add_argument("--git",type=str,default=None,dest="git",
                        help="(Optional) Url to project git repository.")
    package_parser.add_argument("--uploader",type=str,default=None,dest="uploader",
                        help="(Optional) Uploader of the model, for traceability. Follow format 'email,name', email is mandatory, name can be left empty or 'None'.")
    package_parser.add_argument("-n","--name",type=str,default='Unet_Biom3d',dest="model_name",
                        help="(Optional) Name of the model, will serve as archive name.")
    package_parser.add_argument("--packager",action='append',type=str,default=None,dest="packager",
                        help="(Optional) Packager of the model as format: 'name,github_user,affiliation,email,orci', name is mandatory, other field can be written as 'None' and will be ignored. Only if they are not part of the authors.")
    package_parser.add_argument("-c","--cover",type=str,default=None,dest="cover",
                        help="(Optional) Relative path to the cover image, if None, it will be generated with input and output image ")
    package_parser.add_argument("--maintainer", action="append",
                        help="(Optional) GitHub repository maintainers as format: 'name,github_user,affiliation,email,orci', github_user is mandatory, other field can be written as 'None' and will be ignored. Can be repeated")
    package_parser.add_argument("--attach", action="append",
                        help="(Optional) Relative path to a file that will be added to the archive. Can be repeated")
    package_parser.add_argument("--link", action="append",default=None,
                        help="(Optional) IDs of other Bioimage.io ressources (eg:'ilastik/ilastik', 'deepimagej/deepimagej', 'zero/notebook_u-net_3d_zerocostdl4mic'). Can be repeated")
    package_parser.add_argument("--keep_dir", action="store_true",
                        help="(Optional) Whether the temporary folder is kept or not. Giving this parameter will prevent its destruction.")

    args=parser.parse_args()

    if args.command== 'package':
        authors = []
        if args.author:
            for author_str in args.author:
                authors.append(_parse_author_string(author_str))

        cite = []
        if args.cite:
            for cite_str in args.cite:
                cite.append(_parse_cite_string(cite_str))

        uploader = _parse_uploader_string(args.uploader) if args.uploader is not None else None

        maintainers = []
        if args.maintainer:
            for maintainer_str in args.maintainer:
                maintainers.append(_parse_maintainer_string(maintainer_str))

        packagers= []
        if args.packager:
            for packager_str in args.packager:
                packagers.append(_parse_author_string(packager_str))

        training_data = _parse_training_data_string(args.training_data) if args.training_data is not None else None
        parent = _parse_parent_string(args.parent) if args.parent is not None else None
        
        if(args.target == Target.v0x5BIIO):
            package_bioimage_io(args.model,
                                args.img,
                                args.doc,
                                output_folder=args.output_dir,
                                axes=args.axes,
                                model_name=args.model_name,
                                cover=args.cover,
                                author_list=authors,
                                descr=args.descr,
                                license=args.license,
                                cite_list=cite,
                                tags=args.tags,
                                git=args.git,
                                attachments=args.attach,
                                version=args.version,
                                version_comment=args.version_comment,
                                uploader=uploader,
                                maintainers_list=maintainers,
                                packagers_list=packagers,
                                training_data=training_data,
                                parent=parent,
                                links=args.link,
                                run_mode=args.run_mode,
                                config=args.config,
                                keep_dir=args.keep_dir,
                                pred=args.pred,
                                )
    elif args.command=='run-mode':
        kwargs_dict = _parse_kwargs_list(args.kwargs)
        save_run_mode(args.name,args.output,kwargs_dict)
    elif args.command=='config':
        additional_config = _parse_kwargs_list(args.additional_config)
        additional_tolerance = _parse_kwargs_list(args.additional_tolerance)
        tolerance = _parse_tolerance_args(args.tolerance)
        save_config(tolerance,additional_tolerance,additional_config,args.output)
    else:
        raise NotImplementedError("This command does not exist.")