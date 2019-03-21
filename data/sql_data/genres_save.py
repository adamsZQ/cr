from tools.sql_tool import select_all, insert_genre

movie_id_list = []
data_list = select_all()
for data in data_list:
    movie_id = data[0]
    genres = data[4].split('|')
    for genre in genres:
        print(movie_id,genre)
        insert_genre(movie_id,genre)