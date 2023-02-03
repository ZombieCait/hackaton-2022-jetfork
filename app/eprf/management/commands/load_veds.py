import os
import glob
import pandas as pd
from sqlalchemy import create_engine

from django.core.management.base import BaseCommand, CommandError

from eprf.models import VedTranscript
from hackathon import settings


class Command(BaseCommand):
    help = "Загрузка списка ВЕД кодов"


    def handle(self, *args, **kwargs):
        # try:
        folder = os.path.join(settings.BASE_DIR, 'resources/veds/')
        files = glob.glob(folder + '*.csv')
        print(f'Начинаю загрузку таблицы eprf_vedtranscript из папки {folder}. Файлов в очереди: {len(files)}')
        credentials = settings.DATABASES.get('default')
        conn_default = create_engine(
            f'postgresql://{credentials["USER"]}:{credentials["PASSWORD"]}@{credentials["HOST"]}:{credentials["PORT"]}/{credentials["NAME"]}')
        total_count_row = 0
        for file in files:
            df = pd.read_csv(file, sep=';', dtype=str)
            print(f'Загрузка файла {file}.  {len(df)} строк')
            df.rename(columns={
                # 'NAIM1': 'RAZDEL_text',
                'NAIM2': 'GRUPPA_text',
                'NAIM3': 'TOV_POZ_text',
                'NAIM4': 'SUB_POZ_text',
            }, inplace=True)
            df = df[['GRUPPA', 'GRUPPA_text', 'TOV_POZ', 'TOV_POZ_text', 'SUB_POZ', 'SUB_POZ_text', 'VED']]

            start_id = VedTranscript.objects.all().order_by("-id").first()
            if not start_id:
                start_id = 0
            else:
                start_id = start_id.id
            df['id'] = range(start_id + 1, start_id + len(df) + 1)
            df.to_sql('eprf_vedtranscript', if_exists='append', index=False, con=conn_default, chunksize=100000)
            print(f'Выполнено.')
            total_count_row += len(df)
        print(f'Данные загружены. Всего строк: {total_count_row}')
        # except Exception as e:
        #     print(e)
        #     raise CommandError('Произошла ошибка:' + str(e))
