import json

id_list = []
movie_list = []
with open('/path/mv/movie_rating') as f:
    for line in f:
        movie = {}
        line = json.loads(line)
        if line['movie'] not in id_list:
            id_list.append(line['movie'])
            movie['id'] = line['movie']
            movie['critic_rating'] = line['critic_rating']
            movie['audience_rating'] = line['audience_rating']
            movie['director'] = line['director']
            movie['country'] = line['country']
            movie['genres'] = line['genres']
            movie_list.append(movie)

with open('/path/mv/movie_cleaned_v2', 'w') as w:
    for line in movie_list:
        w.write(json.dumps(line))
        w.write('\n')