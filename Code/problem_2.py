import csv

datadict = {}
with open('data_1.csv') as csvfile:
	data = csv.reader(csvfile, delimiter = ',')
	for row in data:
		datadict[row[0]] = row[1]