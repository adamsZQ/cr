import sqlite3

db_file = '../data/wine_data'


def insert_review(id, taster, wine, price, country, province, variety, points):
    conn = sqlite3.connect(db_file)
    try:
        c = conn.cursor()
        c.execute("INSERT INTO review (id, taster, wine, price, country, province, variety, points) VALUES (?,?,?,?,?,?,?,?)", (id, taster, wine, price, country, province, variety, points,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(e)

        conn.commit()
        conn.close()

def insert(table_name, id, stuff_name):
    try:
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        c.execute("INSERT INTO " + table_name + " (id, name) VALUES (?,?)", (id,stuff_name,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(e)

        conn.commit()
        conn.close()


def select_by_name(table_name, stuff_name):
    try:
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        data = c.execute("SELECT id FROM " + table_name + " WHERE name = ?", (stuff_name,))
        data = [x for x in data]
        count = len(data)
        conn.commit()
        conn.close()
        return data
    except Exception as e:
        print(e)

        conn.commit()
        conn.close()


def select_names(table_name):
    try:
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        data = c.execute("SELECT name FROM " + table_name)
        data = [x for x in data]
        conn.commit()
        conn.close()
        return data
    except Exception as e:
        print(e)

        conn.commit()
        conn.close()


def select_ids(table_name):
    try:
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        data = c.execute("SELECT id FROM " + table_name)
        data = [x for x in data]
        conn.commit()
        conn.close()
        return data
    except Exception as e:
        print(e)
        conn.commit()
        conn.close()


if __name__ == '__main__':
    a = select_by_name('country', 'Brazil')
    print(a[0][0])
