import numpy as np
from pathlib import Path
import pandas as pd 
from pprint import pprint 
import yaml 


HERE = Path.cwd()
DATASETS = HERE / "datasets"
FILE_PATH_ADJ = DATASETS / "synapse_count_matrices.xlsx"
FILE_PATH_MERGED = DATASETS / "witv_merged.csv"
COORDS_PATH = DATASETS / "NeuronPosition3D.csv"
BIRTH_PATH = DATASETS / "time_of_birth.xls"
CELLTYPES_PATH = DATASETS / "CellTypes.xlsx"

# ============================================================
# Connectivity matrix processing
# ============================================================
def load_witv(file_path, sheet_name):
    sheet = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    sheet = sheet.T

    post_names = {}
    # for idx, name in enumerate(sheet.iloc[2, 4:]): 
    for idx, name in enumerate(sheet.iloc[2, 4:], start=4): 
        if isinstance(name, str):      
            post_names[name] = idx     

    pre_names = {}
    # for idx, name in enumerate(sheet.iloc[3:, 2]): 
    for idx, name in enumerate(sheet.iloc[3:, 2], start=3): 
        if isinstance(name, str):
            pre_names[name] = idx

    # Use only neurons that appear as both pre- and post-synaptic
    common_names_dict = {}
    for name in pre_names:
        if name in post_names:
            common_names_dict[name] = len(common_names_dict)
        else:
            raise ValueError("pre- and post- are different")

    common_names = []
    for name in pre_names:
        if name in post_names:
            common_names.append(name)
        else:
            raise ValueError("pre- and post- are different")


    N = len(common_names)

    # Now let's fill a matrix of zeros first
    A = np.zeros((N, N), dtype=np.int32)

    for idx_row, pre_name in enumerate(common_names):
        cell_row = pre_names[pre_name]
        for idx_col, post_name in enumerate(common_names):
            cell_col = post_names[post_name]
            val = sheet.iat[cell_row, cell_col]
            if isinstance(val, (int, float)) and not pd.isna(val) and val != 0:
                A[idx_row, idx_col]=1    # binarization  

    # return common_names, N, pre_names, post_names, A
    return common_names, A

# ============================================================
# Coords processing
# ============================================================
def load_neuron_xyz(coords_path, common_names):
    coords_sheet = pd.read_csv(coords_path, header=None)

    mask = coords_sheet.iloc[:, 0].isin(common_names)
    coords_sheet = coords_sheet[mask]
    
    labels = coords_sheet.iloc[:, 0]
    x_coords = coords_sheet.iloc[:, 1]
    y_coords = coords_sheet.iloc[:, 2]
    z_coords = coords_sheet.iloc[:, 3]

    coords_dict = {}
    for label, x, y, z in zip(labels, x_coords, y_coords, z_coords):
        coords_dict[label] = (x, y, z)

    return coords_dict

# ============================================================
# Birth Time processing
# ============================================================
def load_birth_order(birth_path, coords_dict):
    birth_sheet = pd.read_excel(birth_path)

    mask = birth_sheet.iloc[:, 0].isin(coords_dict)  # I am matching against the keys (neuron names)
    birth_sheet = birth_sheet[mask]

    birth_dict = {}
    for name, birth_time in zip(birth_sheet.iloc[:, 0], birth_sheet.iloc[:, 1]):
        birth_dict[name] = birth_time

    return birth_dict


# ============================================================
# Cell types preprocessing
# ============================================================
CELL_TYPE_COL = "cell type"
CELL_SUBTYPE_COL = "cell category"

# Neuronal types in cell types excel
SENSORY_XLS = ["sensory neuron", "sensory"]
MOTOR_XLS = ["motorneuron"]
INTER_XLS = ["interneuron"]

# Neuronal subtypes in cell types excel
SENSORY1_XLS = "SN1"
SENSORY2_XLS = "SN2"
SENSORY3_XLS = "SN3"
SENSORY4_XLS = "SN4"
SENSORY5_XLS = "SN5"
SENSORY6_XLS = "SN6"
MOTOR1_XLS = "head motor neuron"
MOTOR2_XLS = "sublateral motor neuron"
MOTOR3_XLS = "ventral cord motor neuron"
MOTOR4_XLS = "sex-specific neuron"
INTER1_XLS = "layer 1 interneuron"
INTER2_XLS = "layer 2 interneuron"
INTER3_XLS = "layer 3 interneuron"
INTER4_XLS = "category 4 interneuron"
INTER5_XLS = "linker to pharynx"

# Neuronal types in connectome
SENSORY = "S"
SENSORY1 = "S1"
SENSORY2 = "S2"
SENSORY3 = "S3"
SENSORY4 = "S4"
SENSORY5 = "S5"
SENSORY6 = "S6"
MOTOR = "M"
MOTOR1 = "M1"
MOTOR2 = "M2"
MOTOR3 = "M3"
MOTOR4 = "M4"
INTER = "I"
INTER1 = "I1"
INTER2 = "I2"
INTER3 = "I3"
INTER4 = "I4"
INTER5 = "I5"

# Mapping between excel file subtypes and code subtypes
CELL_TYPE_MAPPING = {SENSORY1_XLS: SENSORY1,  # remember: SENSORY1_XLS = "SN1"
                     SENSORY2_XLS: SENSORY2,    # SN2 ...
                     SENSORY3_XLS: SENSORY3,    # etc
                     SENSORY4_XLS: SENSORY4,
                     SENSORY5_XLS: SENSORY5, 
                     SENSORY6_XLS: SENSORY6, 
                     MOTOR1_XLS: MOTOR1, 
                     MOTOR2_XLS: MOTOR2,
                     MOTOR3_XLS: MOTOR3, # MOTOR3_XLS = "ventral cord motor neuron"
                     MOTOR4_XLS: MOTOR4, 
                     INTER1_XLS: INTER1, 
                     INTER2_XLS: INTER2,
                     INTER3_XLS: INTER3,
                     INTER4_XLS: INTER4,  # INTER4_XLS = "category 4 interneuron"
                     INTER5_XLS: INTER5}


# A helper function
def _sheets_filtering(celltypes_path, sheet_name, coords_dict):   # sheet_name="sex-shared") or sheet_name="hermaphrodite specific")
    sheet = pd.read_excel(celltypes_path, sheet_name)
    mask = sheet.iloc[:, 0].isin(coords_dict)
    sheet = sheet[mask]

    return sheet


def _all_types(sheet_filt):
    n_types = {}
    n_subtypes = {}
    for _, row in sheet_filt.iterrows(): # Note: iterrows() returns a tuple: (idx, row_series), so you need idx, row and not only 1 element (like row eg)
        neuron = row.iloc[0]

        # Assign col type
        neuron_type = row[CELL_TYPE_COL]
        
        # Assign col subtype
        neuron_subtype = row[CELL_SUBTYPE_COL]
        
        # Type mapping
        if neuron_subtype not in CELL_TYPE_MAPPING:
            raise ValueError(f"Unknown cell subtype: {neuron_subtype}")
        
        if neuron_type in SENSORY_XLS:  
            neuron_type = SENSORY  
        elif neuron_type in MOTOR_XLS:
            neuron_type = MOTOR
        elif neuron_type in INTER_XLS:
            neuron_type = INTER
        else:
            raise ValueError(f"Unknown cell type: {neuron_type}")

        n_types[neuron] = neuron_type
    
        # Subtype mapping
        n_subtypes[neuron] = CELL_TYPE_MAPPING[neuron_subtype]

    all_types = {}
    for neuron in n_types:
        all_types[neuron] = {
            "type": n_types[neuron],
            "subtype": n_subtypes[neuron],
        }

    return all_types


def load_cell_types(celltypes_path, coords_dict):
    sex_sheet_filt = _sheets_filtering(celltypes_path, sheet_name="sex-shared", coords_dict=coords_dict)
    herm_sheet_filt = _sheets_filtering(celltypes_path, sheet_name="hermaphrodite specific", coords_dict=coords_dict)


    all_types_sex = _all_types(sheet_filt=sex_sheet_filt)
    all_types_herm = _all_types(sheet_filt=herm_sheet_filt)


    # Before merging, let's check for duplicates
    duplicate_neurons = set(all_types_sex) & set(all_types_herm)
    if duplicate_neurons:
        raise ValueError(f"Duplicate neurons found: {sorted(duplicate_neurons)}")

    all_types_info = all_types_sex | all_types_herm


    return all_types_sex, all_types_herm, all_types_info




# ===========================================
# Original adj ordered according to birth time
# ===========================================
def adj_in_birth_t(common_names, A_target, neurons):
    if A_target.shape[0] != A_target.shape[1]:
        raise ValueError(f"A_target must be square, got {A_target.shape}")
    if len(common_names) != A_target.shape[0]:
        raise ValueError(
            f"common_names length ({len(common_names)}) does not match "
            f"A_target size ({A_target.shape[0]})"
        )

    name_to_idx = {}
    for i, neuron in enumerate(common_names):
        name_to_idx[neuron] = i

    # Keep only neurons present in A_target, preserving birth-time order.
    common_names_reordered = []
    idx = []
    for neuron in neurons:
        if neuron not in name_to_idx:
            continue
        common_names_reordered.append(neuron)
        idx.append(name_to_idx[neuron])

    if len(idx) != len(common_names):
        raise ValueError(
            f"Birth-time list covers {len(idx)} of {len(common_names)} neurons in A_target"
        )

    A_target = A_target[np.ix_(idx, idx)]
    # np.fill_diagonal(A_target, 0)

    return common_names_reordered, A_target



# ============================================================
# Merging all in one dataframe
# ============================================================
def processed_witv(celltypes_path, coords_path, birth_path, common_names): 
    coords_dict = load_neuron_xyz(coords_path=coords_path, common_names=common_names)

    _, _, all_types_info = load_cell_types(celltypes_path=celltypes_path, coords_dict=coords_dict)

    birth_dict = load_birth_order(birth_path=birth_path, coords_dict=coords_dict)

    neurons_types = set(all_types_info)
    neurons_coords = set(coords_dict)
    neurons_birth = set(birth_dict)

    # Since elements are unique in dict, this is a sufficient check
    if neurons_types != neurons_coords:
        raise ValueError(
            f"Mismatch between type info and coords: "
            f"{sorted(neurons_types ^ neurons_coords)}"
        )

    if neurons_types != neurons_birth:
        raise ValueError(
            f"Mismatch between type info and birth data: "
            f"{sorted(neurons_types ^ neurons_birth)}"
        )

    ordered_neurons = sorted(birth_dict, key=lambda neuron: int(birth_dict[neuron]))

    all_data_list = []
    for neuron in ordered_neurons:
        all_data_list.append(
            {
                "neuron": neuron,
                "birth_time": int(birth_dict[neuron]),
                "coords": coords_dict[neuron],
                "type": all_types_info[neuron]["type"],
                "subtype": all_types_info[neuron]["subtype"],
            }
        )

    all_data_df = []
    for neuron in ordered_neurons:
        all_data_df.append(
            {
                "neuron": neuron,
                "birth_time": int(birth_dict[neuron]),
                "x": coords_dict[neuron][0],
                "y": coords_dict[neuron][1],
                "z": coords_dict[neuron][2],
                "type": all_types_info[neuron]["type"],
                "subtype": all_types_info[neuron]["subtype"],
            }
        )

    all_data_df = pd.DataFrame(all_data_df)

    return all_data_list, all_data_df 


# To call when needed
def df_to_csv(all_data_df):
    output_path = "witv_merged.csv"
    all_data_df.to_csv(output_path, index=False)
    print(output_path)
    
    
# You can call it like this: 
# df_to_csv(all_data_df)
# ============================================================
# Merging all in one dataframe
# ============================================================










# ============================================================
# ============================================================
# Encoding embeddings
# ============================================================
# ============================================================

# ============================================================
# Extracting lists, dicts from witv_merged.csv
# ============================================================
def from_csv(FILE_PATH_MERGED):
    to_df = pd.read_csv(FILE_PATH_MERGED)

    neurons_list = []
    for neuron in to_df.iloc[:, 0]:
        neurons_list.append(neuron)


    n_features_dict = {}
    for neuron, birth_time, x, y, z, n_type, subtype in zip(
        to_df["neuron"],
        to_df["birth_time"],
        to_df["x"],
        to_df["y"],
        to_df["z"],
        to_df["type"],
        to_df["subtype"],
    ):
        n_features_dict[neuron] = {
            "birth_time": birth_time,
            "x": x, 
            "y": y, 
            "z": z, 
            "type": n_type, 
            "subtype": subtype, 
        }


    class_names_list = sorted(set(to_df["subtype"]))

    return neurons_list, n_features_dict, class_names_list


# ============================================================
# Encoding embeddings from classes 
# ============================================================
def to_class_emb(class_names_list, config):
    embedding_mode = config["embedding_mode"]
    if embedding_mode == "fixed_only":
        D = config["fixed_embedding_size"]
    elif embedding_mode == "dynamic_only":
        D = config["dynamic_embedding_size"]
    else:
        raise ValueError(f"Unknown embedding_mode: {embedding_mode!r}")

    
    if D <= 0:
        raise ValueError(f"D must be > 0, got {D}")
    if D % 2 != 0:
        raise ValueError(f"D must be even for sin/cos class prototypes, got {D}")

    proto = {}
    half = D // 2
    freqs = np.arange(1, half + 1, dtype=np.float64)

    for k, c in enumerate(class_names_list):
        phase = float(k + 1)

        sin_part = np.sin(phase * freqs)
        cos_part = np.cos(phase * freqs)

        v = np.concatenate([sin_part, cos_part]).astype(np.float64)

        norm = np.linalg.norm(v)
        if norm == 0.0:
            raise ValueError(f"Zero-norm prototype for class '{c}' with D={D}")

        v = v / norm
        proto[c] = v.reshape(1, -1)

    return proto




# ============================================================
# Encoding embeddings from class raw 
# ============================================================
def to_class_emb_raw(class_names_list, config):
    embedding_mode = config["embedding_mode"]
    if embedding_mode == "fixed_only":
        D = config["fixed_embedding_size"]
    elif embedding_mode == "dynamic_only":
        D = config["dynamic_embedding_size"]
    else:
        raise ValueError(f"Unknown embedding_mode: {embedding_mode!r}")

    
    if D <= 0:
        raise ValueError(f"D must be > 0, got {D}")
    if D % 2 != 0:
        raise ValueError(f"D must be even for sin/cos class prototypes, got {D}")

    proto = {}
    half = D // 2
    freqs = np.arange(1, half + 1, dtype=np.float64)

    for k, c in enumerate(class_names_list):
        phase = float(k + 1)

        sin_part = np.sin(phase * freqs)
        cos_part = np.cos(phase * freqs)

        v = np.concatenate([sin_part, cos_part]).astype(np.float64)

        # norm = np.linalg.norm(v)
        # if norm == 0.0:
        #     raise ValueError(f"Zero-norm prototype for class '{c}' with D={D}")

        # v = v / norm

        
        proto[c] = v.reshape(1, -1)

    return proto




# ============================================================
# Assigning emb to every neuron (not used)
# ============================================================
def to_n_emb(class_names_list, config, neurons_list, n_features_dict):
    proto = to_class_emb(class_names_list, config)

    n_emb_dict = {}

    if config["birth_noise"]:
        birth_range = float(config["node_emb_birth_uniform_range"])
        if birth_range < 0:
            raise ValueError(f"node_emb_birth_uniform_range must be >= 0, got {birth_range}")

        rng = np.random.default_rng(config["seed"])
    
        for neuron in neurons_list:
            subtype = n_features_dict[neuron]["subtype"]
            base = proto[subtype].copy()
            noise = rng.uniform(
                low=-birth_range,
                high=birth_range,
                size=base.shape,
            ).astype(np.float64)
            n_emb_dict[neuron] = base + noise

    else:
        for neuron in neurons_list:
            subtype = n_features_dict[neuron]["subtype"]
            n_emb_dict[neuron] = proto[subtype].copy() 


    return n_emb_dict 

def to_n_emb_raw(class_names_list, config, neurons_list, n_features_dict):
    proto = to_class_emb_raw(class_names_list, config)

    n_emb_dict = {}

    # if config["birth_noise"]:
    #     birth_range = float(config["node_emb_birth_uniform_range"])
    #     if birth_range < 0:
    #         raise ValueError(f"node_emb_birth_uniform_range must be >= 0, got {birth_range}")

    #     rng = np.random.default_rng(config["seed"])
    
    #     for neuron in neurons_list:
    #         subtype = n_features_dict[neuron]["subtype"]
    #         base = proto[subtype].copy()
    #         noise = rng.uniform(
    #             low=-birth_range,
    #             high=birth_range,
    #             size=base.shape,
    #         ).astype(np.float64)
    #         n_emb_dict[neuron] = base + noise

    # else:
    #     for neuron in neurons_list:
    #         subtype = n_features_dict[neuron]["subtype"]
    #         n_emb_dict[neuron] = proto[subtype].copy() 

    for neuron in neurons_list:
        subtype = n_features_dict[neuron]["subtype"]
        n_emb_dict[neuron] = proto[subtype].copy()

    return n_emb_dict 



def processed_coords(coords_dict, config):
    labels = list(coords_dict.keys())
    coords = np.array([coords_dict[label] for label in labels], dtype=float)

    # Bad!
    if config["norm_and_stand"]:
        row_norms = np.linalg.norm(coords, axis=1, keepdims=True)

        if np.any(row_norms == 0.0):
            zero_norm_labels = [
                labels[i]
                for i in np.where(row_norms[:, 0] == 0.0)[0]
            ]
            raise ValueError(f"Zero-norm coordinate rows for labels: {zero_norm_labels}")

        row_normalized = coords / row_norms

        column_means = row_normalized.mean(axis=0, keepdims=True)
        column_stds = row_normalized.std(axis=0, keepdims=True)

        if np.any(column_stds == 0.0):
            zero_std_columns = np.where(column_stds[0] == 0.0)[0].tolist()
            raise ValueError(f"Zero standard deviation in coordinate columns: {zero_std_columns}")

        processed = (row_normalized - column_means) / column_stds

    elif config["stand_coords"]:
        column_means = coords.mean(axis=0, keepdims=True)
        column_stds = coords.std(axis=0, keepdims=True)

        if np.any(column_stds == 0.0):
            zero_std_columns = np.where(column_stds[0] == 0.0)[0].tolist()
            raise ValueError(f"Zero standard deviation in coordinate columns: {zero_std_columns}")

        processed = (coords - column_means) / column_stds

    elif config["process_all_feats"]:
        processed = coords
    else:
        raise ValueError(
            "Coordinate preprocessing is undefined: set exactly one of "
            "norm_and_stand, stand_coords, or process_all_feats to True."
        )
    
    processed_coords_dict = {}
    for label, values in zip(labels, processed):
        processed_coords_dict[label] = values

    return processed_coords_dict




































_NEURON_TO_CELLTYPE_SPECIAL = {
    'CEPDL': 'CEP', 'CEPDR': 'CEP', 'CEPVL': 'CEP', 'CEPVR': 'CEP',
    'IL1DL': 'IL1', 'IL1DR': 'IL1', 'IL1L': 'IL1', 'IL1R': 'IL1',
    'IL1VL': 'IL1', 'IL1VR': 'IL1',
    'IL2DL': 'IL2_DV', 'IL2DR': 'IL2_DV', 'IL2L': 'IL2_LR', 'IL2R': 'IL2_LR',
    'IL2VL': 'IL2_DV', 'IL2VR': 'IL2_DV',
    'OLQDL': 'OLQ', 'OLQDR': 'OLQ', 'OLQVL': 'OLQ', 'OLQVR': 'OLQ',
    'OLLL': 'OLL', 'OLLR': 'OLL',
    'RMDDL': 'RMD_DV', 'RMDDR': 'RMD_DV', 'RMDL': 'RMD_LR', 'RMDR': 'RMD_LR',
    'RMDVL': 'RMD_DV', 'RMDVR': 'RMD_DV',
    'RMED': 'RME_DV', 'RMEV': 'RME_DV', 'RMEL': 'RME_LR', 'RMER': 'RME_LR',
    'RMFL': 'RMF', 'RMFR': 'RMF', 'RMGL': 'RMG', 'RMGR': 'RMG',
    'RMHL': 'RMH', 'RMHR': 'RMH',
    'SAADL': 'SAA', 'SAADR': 'SAA', 'SAAVL': 'SAA', 'SAAVR': 'SAA',
    'SABVL': 'SAB', 'SABVR': 'SAB', 'SABD': 'SAB',
    'SIADL': 'SIA', 'SIADR': 'SIA', 'SIAVL': 'SIA', 'SIAVR': 'SIA',
    'SIBDL': 'SIB', 'SIBDR': 'SIB', 'SIBVL': 'SIB', 'SIBVR': 'SIB',
    'SMBDL': 'SMB', 'SMBDR': 'SMB', 'SMBVL': 'SMB', 'SMBVR': 'SMB',
    'SMDDL': 'SMD', 'SMDDR': 'SMD', 'SMDVL': 'SMD', 'SMDVR': 'SMD',
    'URADL': 'URA', 'URADR': 'URA', 'URAVL': 'URA', 'URAVR': 'URA',
    'URBL': 'URB', 'URBR': 'URB', 'URXL': 'URX', 'URXR': 'URX',
    'URYDL': 'URY', 'URYDR': 'URY', 'URYVL': 'URY', 'URYVR': 'URY',
    'HSNL': 'HSN', 'HSNR': 'HSN', 'RIVL': 'RIV', 'RIVR': 'RIV',
    'SDQL': 'SDQ', 'SDQR': 'SDQ', 'BAGL': 'BAG', 'BAGR': 'BAG',
    'BDUL': 'BDU', 'BDUR': 'BDU', 'FLPL': 'FLP', 'FLPR': 'FLP',
    'PLML': 'PLM', 'PLMR': 'PLM', 'PLNL': 'PLN', 'PLNR': 'PLN',
    'PVDL': 'PVD', 'PVDR': 'PVD', 'PDEL': 'PDE', 'PDER': 'PDE',
    'PHAL': 'PHA', 'PHAR': 'PHA', 'PHBL': 'PHB', 'PHBR': 'PHB',
    'PHCL': 'PHC', 'PHCR': 'PHC', 'ALML': 'ALM', 'ALMR': 'ALM',
    'ALNL': 'ALN', 'ALNR': 'ALN',
    'AWCL': 'AWC_ON', 'AWCR': 'AWC_OFF',
    'DB1': 'DB01', 'VB1': 'VB01', 'VB2': 'VB02',
    'DD1': 'VD_DD', 'DD2': 'VD_DD', 'DD3': 'VD_DD', 'DD4': 'VD_DD',
    'DD5': 'VD_DD', 'DD6': 'VD_DD',
    'VD1': 'VD_DD', 'VD2': 'VD_DD', 'VD3': 'VD_DD', 'VD4': 'VD_DD',
    'VD5': 'VD_DD', 'VD6': 'VD_DD', 'VD7': 'VD_DD', 'VD8': 'VD_DD',
    'VD9': 'VD_DD', 'VD10': 'VD_DD', 'VD11': 'VD_DD', 'VD12': 'VD_DD',
    'VD13': 'VD_DD',
}


def neuron_to_celltype(name, cell_types):
    """Map individual neuron name to CeNGEN cell type column."""
    if name in _NEURON_TO_CELLTYPE_SPECIAL:
        cell_type = _NEURON_TO_CELLTYPE_SPECIAL[name]
        if cell_type not in cell_types:
            raise KeyError(
                f"Special mapping for neuron '{name}' gives '{cell_type}', "
                f"but that column is not in transcript.csv."
            )
        return cell_type

    if name in cell_types:
        return name

    if len(name) > 1 and name[-1] in ("L", "R"):
        base = name[:-1]
        if base in cell_types:
            return base

    match = _re.match(r"^([A-Z]+)(\d+)$", name)
    if match:
        prefix = match.group(1)
        number = match.group(2)

        if prefix in cell_types:
            return prefix

        padded = f"{prefix}{int(number):02d}"
        if padded in cell_types:
            return padded

    raise KeyError(f"Cannot map neuron '{name}' to any CeNGEN cell type column.")


def load_gene_features(transc_path, genes, neurons):
    """
    Load transcriptomic fixed features.

    Returns
    -------
    features_dict : dict
        neuron -> np.ndarray of shape (1, len(genes))
    """
    genes = list(genes)

    if len(genes) != len(set(genes)):
        raise ValueError(f"Duplicate genes requested: {genes}")

    transc_df = pd.read_csv(transc_path)

    if "gene_name" not in transc_df.columns:
        raise KeyError("transcript.csv must contain a 'gene_name' column.")

    cell_type_cols = transc_df.columns[3:].tolist()
    cell_types = set(cell_type_cols)

    genes_df = transc_df[transc_df["gene_name"].isin(genes)].set_index("gene_name")

    missing = set(genes) - set(genes_df.index)
    if missing:
        raise KeyError(f"Genes not found in transcript.csv: {sorted(missing)}")

    duplicated = genes_df.index[genes_df.index.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"Duplicate gene_name rows in transcript.csv: {duplicated}")

    genes_df = genes_df.loc[genes]

    expr = genes_df[cell_type_cols].astype(np.float64)

    mean = expr.mean(axis=1)
    std = expr.std(axis=1)

    zero_std_genes = std[std == 0.0].index.tolist()
    if zero_std_genes:
        raise ValueError(
            f"Zero standard deviation across cell types for genes: {zero_std_genes}"
        )
    
    processed_expr = expr.sub(mean, axis=0).div(std, axis=0)

    processed_genes_dict = {}
    for neuron in neurons:
        cell_type = neuron_to_celltype(neuron, cell_types)

        if cell_type not in processed_expr.columns:
            raise KeyError(
                f"Cell type '{cell_type}' for neuron '{neuron}' is not a transcriptomics column"
            )

        processed_genes_dict[neuron] = (
            processed_expr[cell_type]
            .to_numpy(dtype=np.float64)
            .reshape(-1)
        )

    return processed_genes_dict