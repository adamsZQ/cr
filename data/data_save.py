import json

from tools.sqlite_tool import select_by_name, insert, insert_review

file_name = 'wine_data_v3.json'
with open(file_name, "r", encoding='utf-8') as f:
    file_contents = f.readlines()

i = j = k = l = 1

taster = []
wine = []
country = []
province = []
variety = []
price = []

for content in file_contents:
    data_json = json.loads(content)
    print(data_json)
    taster = data_json['taster_name']
    wine = data_json['winery']
    country = data_json['country']
    province = data_json['province']
    variety = data_json['variety']
    price = data_json['price']
    points = data_json['points']

    data_taster = select_by_name('taster', taster)[0][0]
    data_wine = select_by_name('wine', wine)[0][0]
    data_country = select_by_name('country', country)[0][0]
    data_province = select_by_name('province', province)[0][0]
    data_variety = select_by_name('variety', variety)[0][0]
    data_price = select_by_name('price', price)[0][0]

    insert_review(i, data_taster, data_wine, data_price, data_country, data_province, data_variety, points)
    i = i + 1




