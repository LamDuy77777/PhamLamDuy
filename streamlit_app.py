import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
from tqdm import tqdm
import pickle
import xgboost as xgb

# Tắt thông báo lỗi từ RDKit
RDLogger.DisableLog('rdApp.*')

# Hàm chuẩn hóa SMILES
def standardize_smiles(batch):
    uc = rdMolStandardize.Uncharger()
    md = rdMolStandardize.MetalDisconnector()
    te = rdMolStandardize.TautomerEnumerator()

    standardized_list = []
    for smi in tqdm(batch, desc='Processing . . .'):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                Chem.SanitizeMol(mol, sanitizeOps=(Chem.SANITIZE_ALL ^ Chem.SANITIZE_CLEANUP ^ Chem.SANITIZE_PROPERTIES))
                cleanup = rdMolStandardize.Cleanup(mol)
                normalized = rdMolStandardize.Normalize(cleanup)
                uncharged = uc.uncharge(normalized)
                fragment = uc.uncharge(rdMolStandardize.FragmentParent(uncharged))
                ionized = rdMolStandardize.Reionize(fragment)
                disconnected = md.Disconnect(ionized)
                tautomer = te.Canonicalize(disconnected)
                smiles = Chem.MolToSmiles(tautomer, isomericSmiles=False, canonical=True)
                standardized_list.append(smiles)
            else:
                standardized_list.append(None)
                st.write(f"Invalid SMILES: {smi}")
        except Exception as e:
            st.write(f"An error occurred with SMILES {smi}: {str(e)}")
            standardized_list.append(None)
    return standardized_list

# Hàm chuyển đổi SMILES thành đặc trưng (Morgan fingerprint)
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return np.array(fp)
    else:
        return np.zeros(2048)

# Hàm chuyển đổi SMILES thành ECFP4
def smiles_to_ecfp4(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return fp

# Hàm tính Tanimoto Distance
def tanimoto_distance(fp1, fp2):
    return 1 - DataStructs.TanimotoSimilarity(fp1, fp2)

# Lớp AD để tính toán điểm SDC
class AD:
    def __init__(self, train_data, nBits=2048, radius=2):
        self.train_data = train_data
        self.nBits = nBits
        self.radius = radius
        self.train_fps = None

    def fit(self):
        self.train_fps = []
        invalid_smiles = []
        for smiles in tqdm(self.train_data, desc="Processing training SMILES"):
            fp = smiles_to_ecfp4(smiles, self.radius, self.nBits)
            if fp is not None:
                self.train_fps.append(fp)
            else:
                invalid_smiles.append(smiles)
        if invalid_smiles:
            st.write(f"Found {len(invalid_smiles)} invalid SMILES in training data: {invalid_smiles}")

    def get_score(self, smiles):
        test_fp = smiles_to_ecfp4(smiles, self.radius, self.nBits)
        if test_fp is None:
            return np.nan
        sdc = 0.0
        for train_fp in self.train_fps:
            td = tanimoto_distance(test_fp, train_fp)
            if td >= 1:
                continue
            exponent = -3 * td / (1 - td)
            sdc += np.exp(exponent)
        return sdc if sdc > 0 else np.nan

# Hàm tính SDC cho tập huấn luyện bằng leave-one-out
def calculate_sdc_loo(train_fps):
    sdc_loo = []
    for i in tqdm(range(len(train_fps)), desc="Calculating leave-one-out SDC"):
        test_fp = train_fps[i]
        other_fps = train_fps[:i] + train_fps[i+1:]
        sdc = 0.0
        for train_fp in other_fps:
            td = tanimoto_distance(test_fp, train_fp)
            if td >= 1:
                continue
            exponent = -3 * td / (1 - td)
            sdc += np.exp(exponent)
        sdc_loo.append(sdc if sdc > 0 else np.nan)
    return sdc_loo

# Tải mô hình XGBoost
@st.cache_resource
def load_model():
    with open('xgboost_binary_10nM.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Giao diện ứng dụng
st.title("Dự đoán với Mô hình XGBoost và Đánh giá Miền Ứng Dụng")

# Tải tập huấn luyện
train_file = st.file_uploader("Tải lên file CSV chứa tập huấn luyện (cột 'standardized' và 'Target1')", type=["csv"])
if train_file:
    data_train = pd.read_csv(train_file)
    data_train = data_train.drop_duplicates(subset=['standardized'])
    st.write(f"Số SMILES trong tập huấn luyện sau khi loại bỏ trùng lặp: {len(data_train)}")

    # Khởi tạo và fit lớp AD
    ad = AD(train_data=data_train['standardized'].to_list())
    ad.fit()

    # Tính SDC cho tập huấn luyện bằng leave-one-out
    train_fps = [smiles_to_ecfp4(s) for s in data_train['standardized'] if smiles_to_ecfp4(s) is not None]
    sdc_loo = calculate_sdc_loo(train_fps)
    sdc_loo = [s for s in sdc_loo if not np.isnan(s)]  # Loại bỏ NaN

    # Chọn ngưỡng SDC: giá trị nhỏ nhất trên tập huấn luyện
    sdc_threshold = np.min(sdc_loo)
    st.write(f"Ngưỡng SDC được chọn: {sdc_threshold}")

# Chọn cách nhập SMILES cho dự đoán
input_method = st.radio("Chọn cách nhập SMILES cho dự đoán:", ("Nhập thủ công", "Tải lên file CSV"))

if input_method == "Nhập thủ công":
    smiles_input = st.text_area("Nhập SMILES (mỗi SMILES trên một dòng):")
    if st.button("Dự đoán") and smiles_input:
        smiles_list = smiles_input.split('\n')
        standardized_smiles = standardize_smiles(smiles_list)
        
        # Lọc SMILES hợp lệ
        valid_smiles = [smi for smi in standardized_smiles if smi is not None]
        if valid_smiles:
            # Chuyển đổi SMILES thành đặc trưng
            features = [smiles_to_features(smi) for smi in valid_smiles]
            model = load_model()
            predictions = model.predict(np.array(features))
            
            # Tính SDC cho các SMILES hợp lệ
            sdc_scores = [ad.get_score(smi) for smi in valid_smiles]
            within_ad = [1 if score >= sdc_threshold else 0 for score in sdc_scores]
            applicability_domain = ["Reliable" if x else "Unreliable" for x in within_ad]
            
            # Tạo DataFrame cho kết quả
            result_df = pd.DataFrame({
                'SMILES đã chuẩn hóa': valid_smiles,
                'Dự đoán': predictions,
                'SDC': sdc_scores,
                'Applicability_Domain': applicability_domain
            })
            st.write("Kết quả dự đoán:", result_df)
        else:
            st.write("Không có SMILES hợp lệ để dự đoán.")

else:
    uploaded_file = st.file_uploader("data_AD_classification_streamlit", type=["csv"])
    if uploaded_file and st.button("Dự đoán"):
        df = pd.read_csv(uploaded_file)
        if 'SMILES' in df.columns:
            smiles_list = df['SMILES'].tolist()
            standardized_smiles = standardize_smiles(smiles_list)
            df['Standardized_SMILES'] = standardized_smiles
            
            # Lọc SMILES hợp lệ
            valid_indices = [i for i, smi in enumerate(standardized_smiles) if smi is not None]
            if valid_indices:
                valid_smiles = [standardized_smiles[i] for i in valid_indices]
                features = [smiles_to_features(smi) for smi in valid_smiles]
                model = load_model()
                predictions = model.predict(np.array(features))
                
                # Tính SDC cho các SMILES hợp lệ
                sdc_scores = [ad.get_score(smi) for smi in valid_smiles]
                within_ad = [1 if score >= sdc_threshold else 0 for score in sdc_scores]
                applicability_domain = ["Reliable" if x else "Unreliable" for x in within_ad]
                
                # Thêm cột dự đoán và AD vào DataFrame
                df.loc[valid_indices, 'Prediction'] = predictions
                df.loc[valid_indices, 'SDC'] = sdc_scores
                df.loc[valid_indices, 'Applicability_Domain'] = applicability_domain
                st.write("Dữ liệu với dự đoán và đánh giá AD:", df[['SMILES', 'Standardized_SMILES', 'Prediction', 'SDC', 'Applicability_Domain']])
            else:
                st.write("Không có SMILES hợp lệ để dự đoán.")
        else:
            st.write("File CSV phải chứa cột 'SMILES'")
