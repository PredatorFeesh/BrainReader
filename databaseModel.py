import os
import csv
import pred
import pickle
from flask import Blueprint
from py2neo import Graph, Node, Relationship, authenticate

database = Blueprint("database", __name__)


def write_to_db(category, name, catOrTest, posOrneg):
    with open("db_info.log", "r") as info:
        password = info.read()
        authenticate("127.0.0.1:7474", "neo4j", password)
        secure_graph = Graph(r"http://127.0.0.1:7474/")

    data = []

    if catOrTest:
        nCategory = Node("Category", name=category, posneg=posOrneg)
    else:
        nCategory = Node("Test", name=category, personName=name)

    with open(r'csv_files/' + name + '.csv') as csvfile: #opens csv files
        datum = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in datum: #iterates
            if i != 0:
                data.append(row)
            i += 1

    for i in range(len(data)): #goes into each row of the data
        # Check if Node exists
        timeNode = secure_graph.data("MATCH (n:Time) WHERE n.timestamp=" + str(i) + " RETURN n")
        if not timeNode:
            timeNode = Node('Time', timestamp=int(i))
        person_time = Relationship(nCategory, 'AtTime', timeNode)
        secure_graph.create(person_time)
        for k in range(2, 6): #looks at relevent channels and adds it to db
            channel = Node('Channel', channelNum=k-1, value=float(data[i][k]))
            person_channel = Relationship(nCategory, category + "'s " + str(k - 1) + ' Channel', channel)
            time_channel = Relationship(timeNode, 'Channel ' + str(k - 1) + " at time", channel)
            secure_graph.create(person_channel)
            secure_graph.create(time_channel)


def to_learning(name):
    cm = pred.class_manager()
    with open("db_info.log", "r") as info:
        password = info.read()
        authenticate("127.0.0.1:7474", "neo4j", password)
        secure_graph = Graph(r"http://127.0.0.1:7474/")
    ch1_data = []
    for z in range(1, 5):
        for i in secure_graph.data("MATCH (n:Category) RETURN n.name"):
            ch1_data.extend(secure_graph.data("MATCH (n:Category {name:'"+ name +"'}) --> (t:Time) --> (a:Channel {channelNum:"+str(z)+"}) WHERE t.timestamp < 2000 RETURN a.value"))
    ch1_data.reverse()
    ch1_data = [i["a.value"] for i in ch1_data]
    ch1_label = []
    for k in range(0, 4):
        ch1_label.extend([k for i in range(2000)])
    ch1 = pred.channel()
    ch1.set_data(ch1_data, ch1_label)
    cm.append(ch1)
    pickled_cm = pickling_into_string(cm)
    Node("Channel Manager", channelmanager=str(pickled_cm))

def pickling_into_string(obj):
    pickler = open("tobepickled.p", "wb")
    pickle.dump(obj, pickler)
    pickler.close()
    the_pickle_file = open("tobepickled.p", "rb")
    serialized_to_string = the_pickle_file.read()
  #  os.remove(os.path.abspath("tobepickled.p"))
    return serialized_to_string

if __name__ == '__main__':
#    write_to_db("Vacation", "Micah", True, "+")
    to_learning("Vacation")

