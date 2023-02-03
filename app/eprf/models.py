from django.db import models

import re, os
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import fasttext
# import nmslib
import razdel
import scipy as sp
import joblib, pickle
from itertools import combinations
from hackathon import settings

# папка с необходимыми файлами для обучения модели
path = os.path.join(settings.BASE_DIR, 'eprf/ml/')

russian_stopwords = open(os.path.join(path, 'stopwords-ru.txt'), 'r').read().split('\n')
prod_name_clf = joblib.load(os.path.join(path, 'product_group.pkl'))
anomaly_detector = CatBoostClassifier()
anomaly_detector.load_model(os.path.join(path, 'anomaly_detector.model'))
reg_clf = fasttext.load_model(os.path.join(path, 'reglament_predictor.model'))
ved_thes = pd.read_csv(os.path.join(settings.BASE_DIR, 'resources', 'veds', 'ved_dict.csv'), sep=';')

for code_col in ["GRUPPA", "TOV_POZ", "SUB_POZ", "VED", "RAZDEL"]:
    ved_thes.loc[ved_thes[code_col].notna(), code_col] = \
        ved_thes.loc[ved_thes[code_col].notna(), code_col].astype(int).astype(str)
with open(os.path.join(path, 'ved_dict.pickle'), "rb") as handle:
    ved_dict = pickle.load(handle)

pmi_hist = pd.read_csv(os.path.join(path, 'pmi_features.csv'), sep=';')
pmi_hist[pmi_hist.columns[:5]] = pmi_hist[pmi_hist.columns[:5]].astype(str)
# index = nmslib.init(method='napp', space='cosinesimil')
# index.loadIndex(os.path.join(path, 'index_ved'), load_data=True)
indexed_data_dict = joblib.load(os.path.join(path, 'index_map.pkl'))
ft_model_v2 = fasttext.load_model(os.path.join(path, 'fb_model_v2.bin'))


def tokenize_with_razdel(text):
    tokens = [token.text for token in razdel.tokenize(text)]

    return tokens


vectorizer = joblib.load(os.path.join(path, 'vectorizer.pkl'))

IDXS = ["Номер продукции"]
FEATURES = ["Технические регламенты", "Группа продукции", "RAZDEL", "GRUPPA", "TOV_POZ"]

class ZipCode(models.Model):
    """ Зип коды """
    country_code = models.CharField(max_length=4, null=True, blank=True)
    zipcode = models.TextField(null=True, blank=True)
    place = models.TextField(null=True, blank=True)
    state = models.TextField(null=True, blank=True)
    state_code = models.TextField(null=True, blank=True)
    province = models.TextField(null=True, blank=True)
    province_code = models.TextField(null=True, blank=True)
    community = models.TextField(null=True, blank=True)
    community_code = models.TextField(null=True, blank=True)
    latitude = models.TextField(null=True, blank=True)
    longitude = models.TextField(null=True, blank=True)

class DataSet(models.Model):
    """ Юридические лица и их адреса"""
    id = models.AutoField(primary_key=True, blank=True)
    id_without_explode = models.IntegerField(blank=True)
    product_number = models.TextField(null=True, blank=True)
    ved_code_id = models.TextField(null=True, blank=True)
    technical_regulations = models.TextField(null=True, blank=True)
    product_group = models.TextField(null=True, blank=True)
    product_name = models.TextField(null=True, blank=True)
    lab_name = models.TextField(null=True, blank=True)
    requester_name = models.TextField(null=True, blank=True)
    requester_address = models.TextField(null=True, blank=True)
    requester_zipcode = models.CharField(max_length=10, null=True, blank=True)
    producer_name = models.TextField(null=True, blank=True)
    producer_country = models.TextField(null=True, blank=True)
    producer_address = models.TextField(null=True, blank=True)

    producer_zipcode = models.CharField(max_length=10, null=True, blank=True)
    producer_country_code = models.TextField(null=True, blank=True)
    requester_country_code = models.TextField(null=True, blank=True)
    requester_latitude = models.TextField(null=True, blank=True)
    requester_longitude = models.TextField(null=True, blank=True)
    producer_latitude = models.TextField(null=True, blank=True)
    producer_longitude = models.TextField(null=True, blank=True)
    GRUPPA = models.TextField(null=True, blank=True)
    GRUPPA_text = models.TextField(null=True, blank=True) # NAIM2
    TOV_POZ =models.TextField(null=True, blank=True)
    TOV_POZ_text = models.TextField(null=True, blank=True) # NAIM3
    SUB_POZ = models.TextField(null=True, blank=True)
    SUB_POZ_text = models.TextField(null=True, blank=True) # NAIM4
    outlier = models.IntegerField(default=None, null=True, blank=True)


class VedTranscript(models.Model):
    """ расшифровка ВЭД
    00 00 000000
    ГР ПЗ СУБПОЗ = ВЭД
    Гр - группа
    ПЗ - товарная позиция
    СУБПОЗ - суб позиция
    """
    GRUPPA = models.CharField(max_length=2)
    GRUPPA_text = models.TextField(null=True, blank=True)
    TOV_POZ = models.CharField(max_length=2, null=True, blank=True)
    TOV_POZ_text = models.TextField(null=True, blank=True)
    SUB_POZ = models.CharField(max_length=6, null=True, blank=True)
    SUB_POZ_text = models.TextField(null=True, blank=True)
    VED = models.CharField(max_length=10, null=True, blank=True)



map_path = os.path.join(settings.BASE_DIR, 'resources/datasets/', 'db_file.csv')
df_map = pd.read_csv(map_path, sep=";").drop("Unnamed: 0", axis=1)
df_map = pd.concat([df_map[df_map['outlier']==1].sample(3000), df_map[df_map['outlier']==0].sample(7000)])
df_map[["product_number", "GRUPPA", "TOV_POZ", "SUB_POZ"]] = \
    df_map[["product_number", "GRUPPA", "TOV_POZ", "SUB_POZ"]].astype(str)
df_map["Аномалия"] = df_map["outlier"].replace(0, "Нет").replace(1, "Да")
