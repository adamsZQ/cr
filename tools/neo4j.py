from py2neo import Graph, NodeSelector

graph = Graph(password='123456')
selector = NodeMatcher(graph)
#selector = NodeSelector(graph)
persons = selector.select('Person', age=21)
print(list(persons))
