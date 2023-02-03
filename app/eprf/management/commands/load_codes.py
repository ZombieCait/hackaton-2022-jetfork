import os
import glob
import pandas as pd
from sqlalchemy import create_engine

from django.core.management.base import BaseCommand, CommandError
import csv

from eprf.models import ZipCode
from hackathon import settings


class Command(BaseCommand):
    help = "Загрузка списка кодов и координат"

    def add_arguments(self, parser):
        parser.add_argument('--csv', type=str)

    def handle(self, *args, **kwargs):
        try:
            folder = os.path.join(settings.BASE_DIR, 'resources/zipcodes/')
            files = glob.glob(folder + '*.csv')
            print(f'Начинаю загрузку таблицы eprf_zipcode из папки {folder}. Файлов в очереди: {len(files)}')
            credentials = settings.DATABASES.get('default')
            conn_default = create_engine(
                f'postgresql://{credentials["USER"]}:{credentials["PASSWORD"]}@{credentials["HOST"]}:{credentials["PORT"]}/{credentials["NAME"]}')
            parts = []
            for file in files:
                df = pd.read_csv(file)
                print(f'Загрузка файла {file}.  {len(df)} строк')
                parts.append(df)
            df = pd.concat(parts, axis=0) \
                .drop_duplicates(subset=['zipcode'], keep="first") \
                .reset_index(drop=True)
            print(f"Duples: {df['zipcode'].duplicated().sum()}")
            start_id = ZipCode.objects.all().order_by("-id").first()
            if not start_id:
                start_id = 0
            else:
                start_id = start_id.id
            df['id'] = range(start_id + 1, start_id + len(df) + 1)
            df.to_sql('eprf_zipcode', if_exists='append', index=False, con=conn_default, chunksize=100000)

                # NO WORK:
                # with open(file) as f:
                #     reader = csv.reader(f)
                #     print(reader)
                #     for row in reader[1:]:
                #         print(row)
                #         zipcode, created = ZipCode.objects.get_or_create(
                #             country_code=str(row[0]),
                #             zipcode=str(row[1]),
                #             place=str(row[2]),
                #             state=str(row[3]),
                #             state_code=str(row[4]),
                #             province=str(row[5]),
                #             province_code=str(row[6]),
                #             community=str(row[7]),
                #             community_code=str(row[8]),
                #             latitude=str(row[9]),
                #             longitude=str(row[10]),
                #         )
                #         count_row += created
                #         zipcode.save()

                # WORK SLOW!
                # df = pd.read_csv(file)
                # count_row = 0
                # for i, row in df.iterrows():
                #     zipcode, created = ZipCode.objects.get_or_create(
                #         country_code=row[0],
                #         zipcode=row[1],
                #         place=row[2],
                #         state=row[3],
                #         state_code=row[4],
                #         province=row[5],
                #         province_code=row[6],
                #         community=row[7],
                #         community_code=row[8],
                #         latitude=row[9],
                #         longitude=row[10],
                #     )
                #     count_row += created
                #     zipcode.save()
        except Exception as e:
            print(e)
            raise CommandError('Произошла ошибка:' + str(e))
