import json
import os
import re
import time
from django.utils import timezone
from io import BytesIO
from itertools import combinations


import plotly.express as px
import joblib as joblib
import numpy as np
import razdel as razdel
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import pandas as pd
import json

from pathlib import Path

from eprf.models import ZipCode, DataSet, VedTranscript, russian_stopwords, prod_name_clf, anomaly_detector, reg_clf, \
    ved_thes, ved_dict, pmi_hist, \
    indexed_data_dict, ft_model_v2, vectorizer, tokenize_with_razdel, IDXS, FEATURES, df_map  # , index # TODO: UNCOMMENT
from hackathon import settings


def main_page(request, *args, **kwargs):
    if request.method == 'GET':
        df = pd.read_parquet(os.path.join(settings.BASE_DIR, 'resources', 'dicts.parquet'), engine='fastparquet')
        # category_filters = Category.objects.distinct('code', 'text').filter(removed=False, sub_code__contains='.0')
        # subcategory_filters = Category.objects.distinct('sub_code', 'sub_text').filter(removed=False)
        return render(request, 'index.html', {'product_groups': df['product_group'].dropna().to_list(),
                                              'technical_regulations': df['technical_regulations'].dropna().to_list(),
                                              })
    else:
        start_time = time.time()
        try:
            df = df_check(request.FILES['excelFile'])
        except Exception as e:
            return render(request, 'index.html', {'status': 'error', 'message': str(e)})

        if request.POST.get('report') == 'xlsx':
            with BytesIO() as b:
                with pd.ExcelWriter(b) as writer:
                    df.to_excel(writer, sheet_name=timezone.now().strftime('result_%Y_%m_%d'), index=False)
                filename = timezone.now().strftime('result_%Y_%m_%d.xlsx')
                res = HttpResponse(
                    b.getvalue(),  # Gives the Byte string of the Byte Buffer object
                    content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
                res['Content-Disposition'] = f'attachment; filename={filename}'
                return res
        else:
            return render(request, 'report.html',
                          {'now': timezone.now().strftime('%d %B %Y %H:%M:%S'),
                           'filename': request.FILES['excelFile'].name,
                           'count': len(df), 'time': time.time() - start_time,
                           # **df.sort_values(by=['light', 'Наличие ошибки'], ascending=False).to_dict(orient='split')})
                           **df.drop(['clean_product_name','light'], axis=1).to_dict(orient='split')})


def check_producer(request, *args, **kwargs):
    countries = DataSet.objects.distinct('producer_country').all()
    ved_groups = VedTranscript.objects.distinct('GRUPPA').all()
    labs = DataSet.objects.distinct('lab_name').all()
    # producers = DataSet.objects.distinct('producer_name').all()

    return render(request, 'check_producer.html', {'filterCountries': countries,
                                                   'filterVedGroups': ved_groups,
                                                   'filterLabs': labs})


def load_csv_codes(file):
    ZipCode





def get_map(request, *args, **kwargs):
    # sample = df_map.copy()

    hover_data = df_map[["producer_country", "producer_name", "lab_name", "ved_code_id"]] \
        .drop_duplicates().reset_index(drop=True)

    fig = px.scatter_geo(
        df_map,
        lat=df_map.producer_latitude,
        lon=df_map.producer_longitude,
        color="Аномалия",
        color_discrete_map={"Нет": "green", "Да": "red"},
        hover_data=hover_data,
        # width=1024,
        height=768,
        opacity=.5
    )
    fig.update(layout_coloraxis_showscale=False)
    from plotly.offline import plot
    plot_div = plot(fig,
                    output_type='div')
    return render(request, "check_producer.html", context={"plot_div": plot_div})



def get_detail_sets(request, *args, **kwargs):
    data = json.loads(request.body)
    if data.get('name') == 'filterVedGroups':
        filterVedGroups = VedTranscript.objects.distinct('GRUPPA').filter(GRUPPA=data.get('val')).only('GRUPPA',
                                                                                                       'GRUPPA_text').all()
        return JsonResponse({'status': 'ok', 'result': list(filterVedGroups.values())})
    return JsonResponse({'status': 'false', 'message': 'Словарь не найден!'}, status=500)


def get_count_by_zips(request, *args, **kwargs):
    data = json.loads(request.body)
    if data.get('zips'):
        pass
    else:
        pass


# ----ML----
path = os.path.join(settings.BASE_DIR, 'eprf/ml/')


def delete_stopwords(s):
    return ' '.join(
        [word for word in (re.sub(r'[()\s+]', u' ', s)).split() if word.lower() not in russian_stopwords]).split()


def delete_punctuation(s):
    symbols = [
        '\t', '!', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '\\', '®',
        '/', '~', '«', '\xad', '¯', '°', '`', '±', '²', '³', '·', 'º', '»', ':', ';', '<', '=', '?', '@',
        'É', 'Ó', 'Ö', '×', 'Ø', 'Ü', 'ä', 'é', 'ö', '÷', 'İ', 'Š', '˂', '˚', '̆', 'Ι', 'Λ', '[', '\\', ']', '_', '`',
        '\u200e', '‐', '–', '—', '‘', '’', '“', '”', '•', '…', '‧', '⁰', '₂', '℃', '№', '™',
        'Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ', '↑', '−', '∞', '≤', '\uf0d2' '️', '（', '）', '，', '0', '1', '2', '3', '4', '5', '6', '7',
        '8', '9'
    ]

    return re.sub(r'[{}\s+]'.format(''.join(symbols)), u' ', s.replace('\xad', ' '))


def tokenize_with_razdel(text):
    tokens = [token.text for token in razdel.tokenize(text)]

    return tokens


def get_product_name(data):
    predicts = prod_name_clf.predict(data)
    return predicts


def write_file(data):
    list1 = data[['Номер продукции', 'Общее наименование продукции',
                  'Коды ТН ВЭД ЕАЭС', 'Технические регламенты',
                  'Группа продукции', 'Наличие ошибки']]
    list1.columns = ['Код', 'Общее наименование продукции',
                     'ТН ВЭД ЕАЭС', 'Технические регламенты',
                     'Группа продукции', 'Наличие ошибки']

    list2 = data[['Номер продукции', 'Общее наименование продукции',
                  'Коды ТН ВЭД ЕАЭС_predicted', 'Технические регламенты_predicted',
                  'Группа продукции_predicted']]
    list2.columns = ['Код', 'Общее наименование продукции',
                     'ТН ВЭД ЕАЭС', 'Технические регламенты',
                     'Группа продукции']
    with pd.ExcelWriter('JETFORK_Тесты2.xlsx', engine='xlsxwriter') as writer:
        list1.to_excel(excel_writer=writer, index=False, sheet_name='Тест1')

        list2.to_excel(excel_writer=writer, index=False, sheet_name='Тест2')


def pmi_predict(dataframe, x, y):
    """Calculate PMI for income data based on historical stats.

    Args:
        dataframe_hist - historical PMI stats;
        dataframe - new data to calculation;
        x - column left;
        y - column right.
    Returns:
        (pd.DataFrame)
    """
    dataframe = dataframe.merge(
        pmi_hist[[x, y, f"pmi_{x}/{y}"]].drop_duplicates(),
        on=[x, y], how="left"
    ).fillna({f"pmi_{x}/{y}": -5})
    return dataframe


def add_ved_info(dataframe):
    """ Add VED main categories.

    Args:
        dataframe - income data.
    Returns:
        (pd.DataFrame)
    """
    return dataframe.merge(
        ved_thes[["VED", "RAZDEL", "GRUPPA", "TOV_POZ"]].drop_duplicates() \
            .rename(columns={"VED": "Коды ТН ВЭД ЕАЭС"}) \
            .astype(str),
        on=["Коды ТН ВЭД ЕАЭС"], how="left"
    ).fillna("-1")


def predict_anomaly(dataframe):
    """ Detect outliers in dataframe attributes.

    Args:
        dataframe - income data.
    Returns:
        (pd.DataFrame)
    """
    d = dataframe.copy()

    dataframe = dataframe.dropna(subset=["Коды ТН ВЭД ЕАЭС", "Технические регламенты", "Группа продукции"])
    dataframe["Коды ТН ВЭД ЕАЭС"] = dataframe["Коды ТН ВЭД ЕАЭС"].astype(str) \
        .str.split("; ") \
        .apply(set)
    dataframe["Технические регламенты"] = dataframe["Технические регламенты"].astype(str) \
        .str.split("; ") \
        .apply(set)
    dataframe["Группа продукции"] = dataframe["Группа продукции"].astype(str) \
        .str.split("; ") \
        .apply(set)
    dataframe = dataframe[~(dataframe["Коды ТН ВЭД ЕАЭС"].apply(len) > 6)].reset_index(drop=True)
    dataframe = dataframe[~(dataframe["Группа продукции"].apply(len) > 2)].reset_index(drop=True)

    dataframe = dataframe.explode("Коды ТН ВЭД ЕАЭС") \
        .dropna() \
        .explode("Технические регламенты") \
        .dropna() \
        .explode("Группа продукции") \
        .dropna() \
        .reset_index(drop=True)

    dataframe = add_ved_info(dataframe)
    dataframe = dataframe[IDXS + FEATURES]

    for f_1, f_2 in combinations(FEATURES, 2):
        if f_1 != f_2:
            dataframe = pmi_predict(dataframe, f_1, f_2)

    dataframe["outlier"] = anomaly_detector.predict(dataframe[anomaly_detector.feature_names_])
    d = d.merge(dataframe.groupby(IDXS)["outlier"].max().reset_index(), on=IDXS, how="left")["outlier"].fillna(
        1).astype(int)
    return d


def predict_reg(s):
    """ Predict tech reg based on product name

    Args:
        s - input string.
    Returns:
        (str)
    """
    s = ' '.join(delete_stopwords(delete_punctuation(s)))
    label = reg_clf.predict(s)[0]
    return ved_dict.get(label[0][9:].replace('_', ' '))


# TODO: UNCOMMENT
# def get_ved(data):
#     vectors = np.array([ft_model_v2.get_sentence_vector(text) for text in data['clean_product_name']])
#     neighbours = index.knnQueryBatch(vectors, k=1, num_threads=10)
#     data['index'] = np.array(neighbours)[:, 0].reshape(-1)
#     # data['distance'] = np.array(neighbours)[:, 1].reshape(-1)
#     veds = data['index'].map(indexed_data_dict).apply(lambda x: ''.join(x))
#     return veds


def df_check(file) -> pd.DataFrame:
    """ Прогон датафрейма через модель
    на вход подается бинарный файл
    """
    original_df = None
    file_format = Path(file.name).suffix
    if file_format == '.xlsx':
        original_df = pd.read_excel(file, dtype=str)
    elif file_format == '.csv':
        original_df = pd.read_csv(file, sep=';', dtype=str)
    else:
        raise Exception('Неправильный формат файла. Разрешенные форматы: .csv, .xslx')

    df = pd.DataFrame()
    try:
        df['Номер продукции'] = original_df['Номер продукции']
        df['Общее наименование продукции'] = original_df['Общее наименование продукции'].astype(str)
        df['Группа продукции'] = original_df['Группа продукции'].astype(str)
        df['Коды ТН ВЭД ЕАЭС'] = original_df['Коды ТН ВЭД ЕАЭС'].astype(str)
        df['Технические регламенты'] = original_df['Технические регламенты'].astype(str)
    except:
        raise Exception('Файл содержит неправильный формат, пожалуйста, скачайте шаблон для заполнения данных')

    df['clean_product_name'] = df['Общее наименование продукции'].fillna('') \
        .str.lower() \
        .apply(lambda x: ' '.join(delete_stopwords(delete_punctuation((x)))))
    vectors = vectorizer.transform(df['clean_product_name'])
    vectors_ft = np.array([ft_model_v2.get_sentence_vector(text) for text in df['clean_product_name']])

    df['Наличие ошибки'] = predict_anomaly(df)
    df['Группа продукции_predicted'] = get_product_name(vectors)
    df['Технические регламенты_predicted'] = df["Общее наименование продукции"].fillna("").apply(predict_reg)
    # TODO: UNCOMMENT
    # df['Коды ТН ВЭД ЕАЭС_predicted'] = get_ved(vectors_ft)

    df['light'] = df['Наличие ошибки'].map({1: 3, 0: 1})  # Зеленый - 0, аномалий нет, 3 - красный - аномалии

    return df


def single_check(request, *args, **kwargs):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            inputProductName = data.get('product_name')
            inputProductCode = data.get('category_name')
            inputVED = data.get('VED')
            inputSubcategoryId = data.get('reglament_name')

            query = pd.DataFrame(data={
                'Номер продукции': [1,],
                'Общее наименование продукции': [inputProductName, ],
                'Группа продукции': [inputProductCode, ],
                'Коды ТН ВЭД ЕАЭС': [inputVED, ],
                'Технические регламенты': [inputSubcategoryId, ]
            })

            result = {}
            print(query)

            # query['Наличие ошибки'] = predict_anomaly(query)

            query['clean_product_name'] = query['Общее наименование продукции'].fillna('') \
                .str.lower() \
                .apply(lambda x: ' '.join(delete_stopwords(delete_punctuation((x)))))

            vectors = vectorizer.transform(query['clean_product_name'])
            # vectors_ft = np.array([ft_model_v2.get_sentence_vector(text) for text in query['clean_product_name']])
            query['Группа продукции_predicted'] = get_product_name(vectors)
            query['Технические регламенты_predicted'] = query["Общее наименование продукции"].fillna("").apply(
                predict_reg)
            # TODO: UNCOMMENT
            # query['Коды ТН ВЭД ЕАЭС_predicted'] = get_ved(vectors_ft)


            # query['light'] = data['Наличие ошибки'].map(
            #     {1: 3, 0: 1})  # Зеленый - 0, аномалий нет, 3 - красный - аномалии

            return JsonResponse({'status': 'ok', "result": {
                'Группа продукции': query.loc[0, 'Группа продукции_predicted'],
                'Технические регламенты': query.loc[0, 'Технические регламенты_predicted'],
                # TODO: UNCOMMENT
                # 'Коды ТН ВЭД': query.loc[0, 'Коды ТН ВЭД ЕАЭС_predicted'],
            }})
        except Exception as e:
            print(e)
            return JsonResponse({'status': 'error', 'detail': str(e)}, status=500)
