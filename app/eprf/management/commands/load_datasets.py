import os
import glob
import pandas as pd
from sqlalchemy import create_engine
from django.core.management.base import BaseCommand, CommandError

from eprf.models import DataSet
from hackathon import settings

import sys



class Command(BaseCommand):
    help = "Загрузка датасетов (таблица eprf_dataset)"

    def handle(self, *args, **kwargs):
        try:
            folder = os.path.join(settings.BASE_DIR, 'resources/datasets/')
            files = glob.glob(folder + '*.csv')
            print(files)
            print(f'Начинаю загрузку таблицы eprf_dataset из папки {folder}. Файлов в очереди: {len(files)}')
            # conn_default = create_engine(settings.DB_URI_DEFAULT).connect()
            credentials = settings.DATABASES.get('default')
            # conn_default = create_engine(credentials).connect()
            conn_default = create_engine(
                f'postgresql://{credentials["USER"]}:{credentials["PASSWORD"]}@{credentials["HOST"]}:{credentials["PORT"]}/{credentials["NAME"]}')
            total_count_row = 0
            for file in files:
                df = pd.read_csv(file, sep=';', index_col=0, dtype=str)
                print(f'Загрузка файла {file}.  {len(df)} строк')

                start_id = DataSet.objects.all().order_by("-id").first()
                if not start_id:
                    start_id = 0
                else:
                    start_id = start_id.id
                df['id'] = range(start_id + 1, start_id + len(df) + 1)
                df.rename(columns={
                    'NAIM2': 'GRUPPA_text',
                    'NAIM3': 'TOV_POZ_text',
                    'NAIM4': 'SUB_POZ_text',
                }, inplace=True)
                df.to_sql('eprf_dataset', if_exists='append', index=False, con=conn_default, chunksize=100000)
                print(f'Выполнено.')
                total_count_row += len(df)
            print(f'Данные загружены. Всего строк: {total_count_row}')
        except Exception as e:
            print(e)
            raise CommandError('Произошла ошибка:' + str(e))



#
# class Command(BaseCommand):
#     help = "Загрузка датасетов (таблица eprf_dataset)"
#
#     def handle(self, *args, **kwargs):
#         try:
#             folder = os.path.join(settings.BASE_DIR, 'resources/datasets/')
#             files = glob.glob(folder + '*.xlsx')
#             print(files)
#             print(f'Начинаю загрузку таблицы eprf_dataset из папки {folder}. Файлов в очереди: {len(files)}')
#             # conn_default = create_engine(settings.DB_URI_DEFAULT).connect()
#             credentials = settings.DATABASES.get('default')
#             # conn_default = create_engine(credentials).connect()
#             conn_default = create_engine(
#                 f'postgresql://{credentials["USER"]}:{credentials["PASSWORD"]}@{credentials["HOST"]}:{credentials["PORT"]}/{credentials["NAME"]}')
#             total_count_row = 0
#             for file in files:
#                 df = pd.read_excel(file)
#                 print(f'Загрузка файла {file}.  {len(df)} строк')
#
#                 # Запоминаем уникальные строки
#                 start_id = DataSet.objects.all().order_by("-id_without_explode").first()
#                 if not start_id:
#                     start_id = 0
#                 else:
#                     start_id = start_id.id_without_explode
#                 df.drop_duplicates(inplace=True)
#                 df['id_without_explode'] = range(start_id + 1, start_id + len(df) + 1)
#
#                 df['check_status'] = None
#                 df.rename(columns={
#                     "Номер продукции": "product_number",
#                     "Коды ТН ВЭД ЕАЭС": "ved_code_id",
#                     "Технические регламенты": 'technical_regulations',
#                     "Группа продукции": 'product_group',
#                     "Общее наименование продукции": 'product_name',
#                     "ИЛ": "lab_name",
#                     "Заявитель": 'requester_name',
#                     "Адрес Заявителя": 'requester_address',
#                     "Изготовитель": 'producer_name',
#                     "Страна": 'producer_country',
#                     "Адрес изготовителя": 'producer_address'
#                 }, inplace=True)
#
#                 df['requester_zip'] = df['requester_address'].str.split() \
#                     .apply(lambda x: x[1] if len(x) > 1 else None)
#
#                 df['producer_zip'] = df['producer_address'].str.split() \
#                     .apply(lambda x: x[1] if len(x) > 1 else None)
#
#                 df['product_group'] = df['product_group'].fillna('').astype(str).str.split("; ").apply(lambda x: list(set(x)))
#                 df = df.explode(column='product_group')
#                 df['ved_code_id'] = df['ved_code_id'].fillna('').astype(str).str.split("; ").apply(lambda x: list(set(x)))
#                 df = df.explode(column='ved_code_id')
#                 df['technical_regulations'] = df['technical_regulations'].fillna('').astype(str).str.split("; ").apply(lambda x: list(set(x)))
#                 df = df.explode(column='technical_regulations')
#
#                 start_id = DataSet.objects.all().order_by("-id").first()
#                 if not start_id:
#                     start_id = 0
#                 else:
#                     start_id = start_id.id
#                 df.drop_duplicates(inplace=True)
#                 df['id'] = range(start_id + 1, start_id + len(df) + 1)
#                 df.to_sql('eprf_dataset', if_exists='append', index=False, con=conn_default, chunksize=100000)
#                 print(f'Выполнено.')
#                 total_count_row += len(df)
#             print(f'Данные загружены. Всего строк: {total_count_row}')
#         except Exception as e:
#             print(e)
#             raise CommandError('Произошла ошибка:' + str(e))
