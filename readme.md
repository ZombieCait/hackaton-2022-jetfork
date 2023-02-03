# Solution from the JETFORK team

screencast https://drive.google.com/drive/folders/1yLE6fezHGcHaO9h38WrXzeK6s9WNmmYe?usp=share_link


A solution that we consider based on open source technologies: Postgree SQL is used as a DBMS, Python libraries for machine learning, Django and Plotly are used for data visualization 


The comprehensive solution we developed reveals 3 scenario events:

Scenario 1. Checking the correct filling of the category, TNVED and regulations and identifying errors, within the framework of the scenario, the applicant can check himself, and he can also receive recommendations on this choice to obtain the name of the product.

Scenarios 2. Allow you to increase the number of manufacturers' search queries and display information about them on the map.

Scenario 3. will be useful to the analyst - with the help of a set of dashboards, you can see typical errors, as well as a data registry for stored information


# Pre-launch setup
1. Необходимо загрузить файлы с zip-кодами в формате csv в папку `app/resources/zipcodes`. Формат файла должен быть `zipcodes.ru.csv`, где `ru` - код страны (это важно). Разделитель `,`;
2. Необходимо загрузить данные с датасетом в формате xslx в папку `app/resources/datasets`;
2. Необходимо загрузить словарь расшифровок ВЕД `app/resources/veds`. Разделителем является `;`!


## Run containers
```shell
 docker-compose -f docker-compose.yml up -d --build
```
после старта контейнеров поднимутся:
* база, доступная по 0.0.0.0:5432
* сервис по адресу 0.0.0.0:8000


# Post-launch setup
1. Загрузка данных **датасета** в БД. Убедитесь, что файл или файлы есть в папке `app/resources/datasets`
```shell
docker exec -it hack_app python manage.py load_datasets
```

2. Загрузка **словарей-расшифровок ВЕД** в БД. Убедитесь, что файл или файлы есть в папке `app/resources/veds`
```shell
docker exec -it hack_app python manage.py load_veds
```

