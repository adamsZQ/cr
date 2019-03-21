import json

from tools.sql_tool import insert


def points2index(num, max):
    if num * 5 < max:
        return 0
    elif num * 5 < max * 2:
        return 1
    elif num * 5 < max * 3:
        return 2
    elif num * 5 < max * 4:
        return 3
    elif num * 5 <= max * 5:
        return 4


# id, critic_rating, audience_rating, director, country, genres
with open('/path/mv/movie_cleaned_v2') as f:
    for line in f:
        line = json.loads(line)
        # line['critic_rating'] = points2index(float(line['critic_rating']), 10.0)
        # line['audience_rating'] = points2index(float(line['audience_rating']), 5.0)
        print(line)
        insert(int(line['id']), line['critic_rating'], line['audience_rating'], line['director'], line['country'], line['genres'])