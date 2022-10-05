import requests
import json
from bs4 import BeautifulSoup
import re
import time
headers = {"User-Agent": "Mozilla/5.0"}
url = "https://www.wildberries.ru/catalog/elektronika/noutbuki-i-kompyutery/komplektuyushchie-dlya-pk?sort=popular&page=1&xsubject=3698"
arr_element_json = []
session = requests.Session()
response_text = session.get(url, headers=headers).text
soup = BeautifulSoup(response_text, 'lxml')
for i in range(int(int(re.sub(r"\D", "", soup.find('span', class_="goods-count").text)) / 100) + 1):
    time.sleep(2)
    response_text = session.get(f"https://www.wildberries.ru/catalog/elektronika/noutbuki-i-kompyutery/komplektuyushchie-dlya-pk?sort=popular&page={i}&xsubject=3698",
                                headers=headers).text
    soup = BeautifulSoup(response_text, 'lxml')
    TSearch = soup.find_all('div', class_="product-card j-card-item")
    for element in TSearch:
        if (element.find('span', class_="goods-name").text == "Процессор"):
            continue
        tmp = {"name": element.find('span', class_="goods-name").text,
               "id": element["id"],
               "price": None,
               "old-price": None,
               "sale": None,
               "stars": None
               }

        if (element.find('ins', class_="lower-price")) != None:
            tmp["price"] = re.sub(r"\D", "", element.find('ins', class_="lower-price").text)
        elif (element.find('span', class_="lower-price")) != None:
            tmp["price"] = re.sub(r"\D", "", element.find('span', class_="lower-price").text)
        if (element.find('span', class_="price-old-block")) != None:
            tmp["old-price"] = re.sub(r"\D", "", element.find('span', class_="price-old-block").text)
        if (element.find('span', class_="product-card__sale") != None):
            tmp["sale"] = re.sub(r"\D", "", element.find('span', class_="product-card__sale").text)
        if (element.find('span', class_="product-card__count") != None):
            tmp["stars"] = re.sub(r"\D", "", element.find('span', class_="product-card__count").text)
        arr_element_json.append(tmp)
session.close()


arr_element_json = [json.dumps(el) for el in arr_element_json]
arr_element_json = list(set(arr_element_json))
arr_element_json = [json.loads(el) for el in arr_element_json]

with open("dump.json", "w") as f:
    json.dump(arr_element_json, f)

for i in range(len(arr_element_json)):
    for j in range(len(arr_element_json)):
        if int(arr_element_json[i]["price"]) > int(arr_element_json[j]["price"]):
            tmp = arr_element_json[i]
            arr_element_json[i] = arr_element_json[j]
            arr_element_json[j] = tmp

lowest, highest = map(int, input("Введите наименьшую и наибольшую цену: ").split())
del_flag = True
while del_flag:
    del_flag = False
    for i in range(len(arr_element_json)):
        if lowest < int(arr_element_json[i]["price"]) < highest:
            continue
        else:
            del_flag = True
            del arr_element_json[i]
            break



for element in arr_element_json:
    print(element)
